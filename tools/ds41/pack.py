"""tools/ds41/pack.py - turn an EXL3 DeepSeek V4.1 Flash checkpoint into the ds41 engine pack.

    python tools/ds41/pack.py --src /workspace/model --out /workspace/pack-3bpw

Input: an EXL3 checkpoint in Hugging Face layout (coolbho3k 3bpw, SAGE mixed-bitrate, or ours): routed experts
as EXL3 mul1 tensors (trellis, suh, svh, mul1), everything else as DeepSeek published it (FP8 E4M3 with E8M0
32x32 block scales, BF16, F32), Engram tables under engrams/.

Output (the engine reads only flat text indexes written here, as upstream Strata does: Python writes the layout
once, the engine never re-derives it):

  dense.bin      every non-expert text tensor, raw bytes, each 256 B aligned. Vision tensors are left out.
                 attn.wo_a stays FP8 with its block scales (--wo-a-bf16: dequantized to BF16 exactly as DeepSeek's
                 inference/convert.py does; the engine reads both).
  index.txt      one line per dense tensor: name dtype ndim dims... offset bytes
  experts.bin    one slot per routed expert, layer-major then expert-major. A slot starts on a 4 KiB boundary
                 and its length is a multiple of 4 KiB, so one O_DIRECT read fetches a whole expert.
                 Inside a slot: w1, w3, w2, each as trellis, suh, svh, mul1; each component 256 B aligned.
  experts.txt    one line per expert: layer expert offset bytes w1_bits w3_bits w2_bits, then 12 entries
                 component:offset:bytes (offset inside the slot). Bitrates may differ per expert (SAGE).
  engram.txt     one line per Engram layer: layer rows dim weight_offset scale_offset path (tables are not copied)
  engram_hash.txt, engram_tokenmap.bin   the n-gram hash tables, computed here with DeepSeek's own engram.py
                 (the compressed token map needs the tokenizer's normalizer; the engine needs only integers)
  config.json, tokenizer.json, tokenizer_config.json   copied
  pack_info.txt  written last; "finished 1" marks a complete pack
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import time

import numpy as np

DENSE_ALIGN = 256
COMPONENT_ALIGN = 256
EXPERT_ALIGN = 4096
DTYPE = {"F8_E4M3": "f8e4m3", "F8_E8M0": "e8m0", "BF16": "bf16", "F32": "f32", "F16": "f16",
         "I32": "i32", "I16": "i16", "I8": "i8", "U8": "u8"}
PROJ = ("w1", "w3", "w2")
PARTS = ("trellis", "suh", "svh", "mul1")
SKIP_PREFIX = ("vision.", "aligner.", "image_")


def align(x, a):
    return (x + a - 1) // a * a


class Source:
    """Safetensors headers of a checkpoint, and pread access to tensor bytes."""

    def __init__(self, src):
        self.src = src
        idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
        self.entries, self.fds = {}, {}
        for fname in sorted(set(idx.values())):
            path = os.path.join(src, fname)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            self.fds[fname] = os.open(path, os.O_RDONLY)
            for k, v in hdr.items():
                a, b = v["data_offsets"]
                self.entries[k] = (fname, v["dtype"], v["shape"], 8 + n + a, b - a)
        missing = set(idx) - set(self.entries)
        if missing:
            raise SystemExit(f"index names {len(missing)} tensors not found in the shards, e.g. {sorted(missing)[:3]}")

    def read(self, name):
        fname, _, _, off, nb = self.entries[name]
        buf = bytearray(nb)
        got = os.preadv(self.fds[fname], [buf], off)
        if got != nb:
            raise IOError(f"short read for {name}: {got} of {nb}")
        return buf


def bits_of(trellis_shape):
    """EXL3 tile width is 16*K uint16 per tile, 16*K + 8 for a half-integer K."""
    w = trellis_shape[-1]
    return w // 16 if w % 16 == 0 else (w - 8) / 16 + 0.5


def dequant_wo_a(w_bytes, w_shape, s_bytes, s_shape):
    """FP8 E4M3 weight times its E8M0 block scale, rounded to BF16 (inference/convert.py)."""
    import torch
    w = torch.frombuffer(w_bytes, dtype=torch.float8_e4m3fn).view(*w_shape).float()
    s = torch.frombuffer(s_bytes, dtype=torch.float8_e8m0fnu).view(*s_shape).float()
    bo, bi = w_shape[0] // s_shape[0], w_shape[1] // s_shape[1]
    w = w.unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)) * s[:, None, :, None]
    return bytearray(w.flatten(2, 3).flatten(0, 1).bfloat16().contiguous().view(torch.uint8).numpy().tobytes())


def write_dense(srcs, out, wo_a_fp8=True):
    """wo_a_fp8: keep attn.wo_a as FP8 E4M3 with its E8M0 block scales (the engine dequantizes it on the fly, to the
    same BF16 values); else dequantize it to BF16 here, as inference/convert.py does."""
    names = sorted(n for n in srcs.entries
                   if ".ffn.experts." not in n and not n.startswith(SKIP_PREFIX)
                   and (wo_a_fp8 or not n.endswith("wo_a.scale")) and ".engram.embed." not in n)
    off = 0
    with open(os.path.join(out, "dense.bin"), "wb") as fb, open(os.path.join(out, "index.txt"), "w") as fi:
        fi.write("# ds41 dense index v1: name dtype ndim dims... offset bytes\n")
        for n in names:
            _, dt, shape, _, _ = srcs.entries[n]
            data = srcs.read(n)
            if n.endswith("wo_a.weight") and not wo_a_fp8:
                sname = n[:-len("weight")] + "scale"
                data = dequant_wo_a(data, shape, srcs.read(sname), srcs.entries[sname][2])
                dt = "BF16"
            pad = align(off, DENSE_ALIGN) - off
            fb.write(b"\0" * pad)
            off += pad
            fb.write(data)
            fi.write(f"{n} {DTYPE[dt]} {len(shape)} {' '.join(map(str, shape))} {off} {len(data)}\n")
            off += len(data)
    return len(names), off


def expert_layout(srcs, L, e):
    comps, off, bits = [], 0, {}
    for w in PROJ:
        for p in PARTS:
            name = f"layers.{L}.ffn.experts.{e}.{w}.{p}"
            nb = srcs.entries[name][4]
            comps.append((f"{w}.{p}", name, off, nb))
            off = align(off + nb, COMPONENT_ALIGN)
        bits[w] = bits_of(srcs.entries[f"layers.{L}.ffn.experts.{e}.{w}.trellis"][2])
    return comps, align(off, EXPERT_ALIGN), bits


def write_experts(srcs, out, n_layers, n_experts):
    pos = 0
    t0 = time.time()
    with open(os.path.join(out, "experts.bin"), "wb") as fb, open(os.path.join(out, "experts.txt"), "w") as ft:
        ft.write("# ds41 experts index v1: layer expert offset bytes w1_bits w3_bits w2_bits "
                 "then component:offset:bytes x12 (offset inside the slot)\n")
        for L in range(n_layers):
            for e in range(n_experts):
                comps, slot, bits = expert_layout(srcs, L, e)
                buf = bytearray(slot)
                for _, name, coff, nb in comps:
                    buf[coff:coff + nb] = srcs.read(name)
                fb.write(buf)
                fields = " ".join(f"{c}:{coff}:{nb}" for c, _, coff, nb in comps)
                ft.write(f"{L} {e} {pos} {slot} {bits['w1']} {bits['w3']} {bits['w2']} {fields}\n")
                pos += slot
            print(f"layer {L} done, {pos / 2**30:.2f} GiB, {time.time() - t0:.0f} s", flush=True)
    return pos


def read_experts_txt(path):
    rows = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        L, e, off, nb = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        num = lambda s: int(s) if float(s).is_integer() else float(s)
        bits = {"w1": num(f[4]), "w3": num(f[5]), "w2": num(f[6])}
        comps = {}
        for item in f[7:]:
            c, a, b = item.split(":")
            comps[c] = (int(a), int(b))
        rows[(L, e)] = {"offset": off, "bytes": nb, "bits": bits, "components": comps}
    return rows


def write_engram(src, out):
    """The Engram tables stay in their files: engrams/*.safetensors (the 3bpw pack) or the index's shards that hold
    them (SAGE 1.59bpw)."""
    paths = []
    d = os.path.join(src, "engrams")
    if os.path.isdir(d):
        paths += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".safetensors")]
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    paths += [os.path.join(src, f) for f in sorted({f for k, f in idx.items() if k.endswith(".engram.embed.weight")})]
    tables = {}
    for path in map(os.path.abspath, paths):
        if "\n" in path or "\r" in path:
            raise ValueError("Engram paths must not contain newlines")
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        hdr.pop("__metadata__", None)
        for k, v in hdr.items():
            if k.endswith(".engram.embed.weight"):
                L = int(k.split(".")[1])
                s = hdr[k.replace("weight", "scale")]
                tables.setdefault(L, f"{L} {v['shape'][0]} {v['shape'][1]} {8 + n + v['data_offsets'][0]} "
                                     f"{8 + n + s['data_offsets'][0]} {path}")
    lines = [tables[L] for L in sorted(tables)]
    with open(os.path.join(out, "engram.txt"), "w") as f:
        f.write("# ds41 engram tables v1: layer rows dim weight_offset scale_offset path\n")
        f.write("".join(l + "\n" for l in lines))
    return len(lines)


REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ds41", "proto", "ref")


def write_engram_hash(args, tokenizer, out):
    """Export what NgramHashState computes from the tokenizer: compressed token map, primes, offsets,
    multipliers and the pad id. `args` has the inference config's engram_* fields."""
    import sys
    sys.path.insert(0, REF_DIR)
    import engram as EG
    token_map, vocab = EG.build_compressed_token_map(tokenizer)
    if vocab != args.engram_compressed_vocab_size:
        raise SystemExit(f"compressed vocab {vocab} != config {args.engram_compressed_vocab_size}: "
                         "wrong tokenizer, every hash would differ")
    layout = EG.EngramLayout.from_args(args)
    mult = EG.compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab)
    np.array(token_map, dtype="<i4").tofile(os.path.join(out, "engram_tokenmap.bin"))
    with open(os.path.join(out, "engram_hash.txt"), "w") as f:
        f.write("# ds41 engram hash v1. Per engram layer: multipliers [max_ngram], then primes and offsets\n"
                "# [ (max_ngram-1) * n_heads ], ngram size major, head minor. token map: engram_tokenmap.bin\n")
        f.write(f"max_ngram {layout.max_ngram_size}\nn_heads {layout.n_heads}\n")
        f.write(f"pad {token_map[args.engram_pad_id]}\nvocab {vocab}\n")
        f.write("layers " + " ".join(map(str, layout.layer_ids)) + "\n")
        for li, layer in enumerate(layout.primes):
            flat = [p for per in layer for p in per]
            offs = np.cumsum([0, *flat[:-1]])
            f.write(f"multipliers {li} " + " ".join(str(int(m)) for m in mult[li]) + "\n")
            f.write(f"primes {li} " + " ".join(map(str, flat)) + "\n")
            f.write(f"offsets {li} " + " ".join(str(int(o)) for o in offs) + "\n")


def read_engram_hash(d):
    t = {"multipliers": {}, "primes": {}, "offsets": {}}
    for line in open(os.path.join(d, "engram_hash.txt")):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if f[0] in ("max_ngram", "n_heads", "pad", "vocab"):
            t[f[0]] = int(f[1])
        elif f[0] == "layers":
            t["layers"] = [int(x) for x in f[1:]]
        else:
            t[f[0]][int(f[1])] = [int(x) for x in f[2:]]
    t["token_map"] = np.fromfile(os.path.join(d, "engram_tokenmap.bin"), dtype="<i4")
    return t


def engram_hash_reference(ids, t):
    """Plain re-statement of NgramHashState.forward for text-only input: what the C++ engine implements.
    Returns int64 [len(ids), n_engram_layers, (max_ngram-1) * n_heads]."""
    n, nh, nl = t["max_ngram"], t["n_heads"], len(t["layers"])
    comp = [int(t["token_map"][i]) for i in ids]
    out = np.zeros((len(ids), nl, (n - 1) * nh), dtype=np.int64)
    for pos in range(len(ids)):
        toks = [comp[pos - s] if pos - s >= 0 else t["pad"] for s in range(n)]
        for li in range(nl):
            prod = [toks[s] * t["multipliers"][li][s] for s in range(n)]
            rolling = prod[0]
            for i in range(1, n):
                rolling ^= prod[i]
                for h in range(nh):
                    c = (i - 1) * nh + h
                    out[pos, li, c] = rolling % t["primes"][li][c] + t["offsets"][li][c]
    return out


class EngramArgs:
    """The inference config's engram_* fields, read from the checkpoint's HF config.json."""

    def __init__(self, cfg):
        tc = cfg.get("text_config", cfg)
        self.engram_layer_ids = tuple(tc["engram_layer_ids"])
        self.engram_num_embeddings = tuple(tc["engram_num_embeddings"])
        self.engram_max_ngram_size = tc["engram_max_ngram_size"]
        self.engram_vocab_size = tc["engram_vocab_size"]
        self.engram_n_heads = tc["engram_n_heads"]
        self.engram_head_dim = tc["engram_head_dim"]
        self.engram_pad_id = tc.get("engram_pad_token_id", tc.get("engram_pad_id"))
        self.engram_compressed_vocab_size = tc["engram_compressed_vocab_size"]


def build_pack(src, out, n_layers=40, n_experts=384, wo_a_fp8=True, dense_only=False):
    """dense_only: rewrite dense.bin and index.txt of a finished pack; experts.bin and the rest stay."""
    os.makedirs(out, exist_ok=True)
    info = os.path.join(out, "pack_info.txt")
    old = {}
    if dense_only:
        old = dict(l.split(None, 1) for l in open(info).read().splitlines() if l.strip())
        if old.get("finished", "").strip() != "1":
            raise SystemExit("--dense-only needs a finished pack")
    if os.path.exists(info):
        os.remove(info)                  # an old marker must not vouch for a new, half-written pack
    srcs = Source(src)
    n_dense, dense_bytes = write_dense(srcs, out, wo_a_fp8)
    if dense_only:
        with open(info, "w") as f:
            f.write(f"source {os.path.abspath(src)}\nlayers {old['layers']}\nexperts {old['experts']}\n"
                    f"dense_tensors {n_dense}\ndense_bytes {dense_bytes}\nexpert_bytes {old['expert_bytes']}\n"
                    f"engram_layers {old['engram_layers']}\nfinished 1\n")
        return
    expert_bytes = write_experts(srcs, out, n_layers, n_experts)
    n_engram = write_engram(src, out)
    cfg = json.load(open(os.path.join(src, "config.json")))
    if "engram_layer_ids" in cfg.get("text_config", cfg):
        from transformers import AutoTokenizer
        write_engram_hash(EngramArgs(cfg), AutoTokenizer.from_pretrained(src), out)
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(os.path.join(src, f), os.path.join(out, f))
    with open(info, "w") as f:
        f.write(f"source {os.path.abspath(src)}\nlayers {n_layers}\nexperts {n_experts}\n"
                f"dense_tensors {n_dense}\ndense_bytes {dense_bytes}\nexpert_bytes {expert_bytes}\n"
                f"engram_layers {n_engram}\nfinished 1\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--wo-a-bf16", action="store_true", help="dequantize attn.wo_a to BF16 (packs before 2026-10-06)")
    ap.add_argument("--dense-only", action="store_true", help="rewrite only dense.bin and index.txt of a finished pack")
    a = ap.parse_args()
    build_pack(a.src, a.out, a.layers, a.experts, wo_a_fp8=not a.wo_a_bf16, dense_only=a.dense_only)


if __name__ == "__main__":
    main()
