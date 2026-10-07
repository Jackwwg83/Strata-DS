"""DeepSeek V4.1 Flash on one consumer GPU: a slow, faithful prototype forward.

Purpose: run the real model end to end on target-class hardware (one 16-24 GB GPU, 128 GB RAM,
NVMe) to (1) produce real tokens and (2) record real expert routing for cache design. Speed is
not a goal here; Strata-DS's engine is.

The model code is DeepSeek's own ref/model.py, imported unmodified. This file patches only:
  - routed experts: coolbho3k EXL3 3bpw weights, read from the SSD per use (pread into pinned
    buffers, then H2D) and run with exllamav3's EXL3 kernels. Inputs get the same FP8 block
    quantization the official FP4 path applies (--no-expert-act-fp8 turns it off).
  - Engram tables: rows read from the SSD (the official code keeps the 2 x 98 GB tables on GPU).
  - output head: bf16 storage instead of a 2.6 GB fp32 copy; logits still computed in fp32.
  - `kernel`: ref/kernel.py (TileLang) if it imports, else torch_kernels.py.
DSpark/MTP and vision are disabled.

Usage:
  python ds41_proto.py --model-dir /workspace/model --out /workspace/results/run \
      --corpus corpus/docs.jsonl --prompts corpus/prompts.jsonl
  python ds41_proto.py --model-dir ... --out ... --smoke       # one short prompt end to end
"""
import argparse
import json
import os
import queue
import struct
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ref"))
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

ST_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "BF16": torch.bfloat16,
             "F16": torch.float16, "F32": torch.float32, "I16": torch.int16, "I32": torch.int32,
             "I8": torch.int8, "U8": torch.uint8}
H, FFN, N_LAYERS, TOPK = 5120, 2304, 40, 6
SLOT_ALIGN = 256


def select_kernels(choice):
    """Return the kernel module model.py will import, and its name."""
    if choice in ("auto", "tilelang"):
        try:
            import kernel  # ref/kernel.py
            import torch_kernels
            q = torch.zeros(1, 1, 64, 512, dtype=torch.bfloat16, device="cuda")   # the model's real shape
            kv = torch.zeros(1, 4, 512, dtype=torch.bfloat16, device="cuda")
            idx = torch.zeros(1, 1, 4, dtype=torch.int32, device="cuda")
            try:
                kernel.sparse_attn(q, kv, torch.zeros(64, device="cuda", dtype=torch.float32), idx, 0.1)
            except Exception as ex:   # needs 141 KB shared memory per block: Hopper-class GPUs only
                print(f"TileLang sparse_attn unusable here ({str(ex)[:80]}); using the torch version")
                kernel.sparse_attn = torch_kernels.sparse_attn
                return kernel, "tilelang+torch_sparse_attn"
            return kernel, "tilelang"
        except Exception as ex:  # TileLang missing or unsupported on this GPU
            if choice == "tilelang":
                raise
            print(f"TileLang kernels unavailable ({type(ex).__name__}: {ex}); using torch_kernels")
    import torch_kernels
    sys.modules["kernel"] = torch_kernels
    return torch_kernels, "torch"


# ---------------------------------------------------------------- checkpoint access (mmap + pread)

class SafetensorsFile:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        base = 8 + n
        self.entries = {k: (v["dtype"], v["shape"], base + v["data_offsets"][0],
                            v["data_offsets"][1] - v["data_offsets"][0]) for k, v in hdr.items()}
        self.fd = os.open(path, os.O_RDONLY)
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def view(self, name):
        dt, shape, off, nb = self.entries[name]
        t = torch.from_numpy(self.mm[off:off + nb])
        return t.view(ST_DTYPES[dt]).view(shape)


class Checkpoint:
    def __init__(self, model_dir):
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.files = {f: SafetensorsFile(os.path.join(model_dir, f)) for f in sorted(set(idx.values()))}
        self.where = {k: self.files[f] for k, f in idx.items()}

    def get(self, name):
        return self.where[name].view(name)

    def has(self, name):
        return name in self.where


# ---------------------------------------------------------------- routed experts

class ExpertStore:
    """Reads one expert's 12 EXL3 tensors with pread into a pinned slot (worker threads, so the
    SSD sees several requests at once), copies the slot to the GPU, and builds LinearEXL3 views."""
    PARTS = ("trellis", "suh", "svh", "mul1")
    PROJ = (("w1", H, FFN), ("w3", H, FFN), ("w2", FFN, H))

    def __init__(self, ckpt, threads=8, slots=48, pin=True):
        self.ckpt = ckpt
        self.pool = ThreadPoolExecutor(threads)
        # experts may differ in size (SAGE 1.59bpw mixes K per projection): a slot holds the largest one
        n_exp = sum(1 for k in ckpt.where if k.startswith("layers.0.ffn.experts.") and k.endswith(".w1.trellis"))
        self.slot_bytes = max(self._layout(l, e)[1] for l in range(N_LAYERS) for e in range(n_exp))
        self.free = queue.Queue()
        for _ in range(slots):
            self.free.put(torch.empty(self.slot_bytes, dtype=torch.uint8, device="cpu", pin_memory=pin))
        self.bytes_read = 0
        self.read_s = 0.0

    def _layout(self, layer, e):
        """(w, part, dtype, shape, offset in the slot, bytes) of each tensor of expert (layer, e), and its size"""
        layout, off = [], 0
        for w, _, _ in self.PROJ:
            for p in self.PARTS:
                name = f"layers.{layer}.ffn.experts.{e}.{w}.{p}"
                dt, shape, _, nb = self.ckpt.where[name].entries[name]
                layout.append((w, p, dt, shape, off, nb))
                off += (nb + SLOT_ALIGN - 1) // SLOT_ALIGN * SLOT_ALIGN
        return layout, off

    def _read(self, layer, e):
        slot = self.free.get()
        buf = slot.numpy()
        t0 = time.perf_counter()
        layout, _ = self._layout(layer, e)
        for w, p, dt, shape, off, nb in layout:
            name = f"layers.{layer}.ffn.experts.{e}.{w}.{p}"
            f = self.ckpt.where[name]
            foff = f.entries[name][2]
            got = os.preadv(f.fd, [memoryview(buf[off:off + nb])], foff)
            assert got == nb, (name, got, nb)
        return slot, layout, time.perf_counter() - t0

    def stream(self, layer, ids, lookahead=32):
        """Yield (expert_id, (w1, w3, w2)) in order, with reads running `lookahead` ahead."""
        from exllamav3.modules.quant.exl3 import LinearEXL3
        futs = {}
        ids = list(ids)
        for i, e in enumerate(ids[:lookahead]):
            futs[i] = self.pool.submit(self._read, layer, e)
        for i, e in enumerate(ids):
            slot, layout, dt = futs.pop(i).result()
            nxt = i + lookahead
            if nxt < len(ids):
                futs[nxt] = self.pool.submit(self._read, layer, ids[nxt])
            dev = torch.empty(self.slot_bytes, dtype=torch.uint8, device="cuda")
            dev.copy_(slot)                    # synchronous: the slot is free again after this
            self.free.put(slot)
            self.bytes_read += self.slot_bytes
            self.read_s += dt
            t = {}
            for w, p, dts, shape, off, nb in layout:
                t[(w, p)] = dev[off:off + nb].view(ST_DTYPES[dts]).view(shape)
            lins = tuple(LinearEXL3(None, k, n, suh=t[(w, "suh")], svh=t[(w, "svh")],
                                    trellis=t[(w, "trellis")], mul1=t[(w, "mul1")], key=f"L{layer}.E{e}.{w}")
                         for w, k, n in self.PROJ)
            yield e, lins


class RouteLog:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ids, self.w = {}, {}

    def add(self, layer, indices, weights):
        self.ids.setdefault(layer, []).append(indices.to(torch.int16).cpu())
        self.w.setdefault(layer, []).append(weights.to(torch.float16).cpu())

    def stacked(self):
        layers = sorted(self.ids)
        ids = torch.stack([torch.cat(self.ids[l]) for l in layers], dim=1)   # [T, L, k]
        w = torch.stack([torch.cat(self.w[l]) for l in layers], dim=1)
        return ids.numpy(), w.numpy()


class CpuExperts:
    """Routed experts on the CPU with exllamav3's moe_mul1 kernel, the kernel the C++ engine uses.
    Weights are zero-copy views of the mmap'ed checkpoint; a layer registers on first use, unswizzled."""

    def __init__(self, ckpt, threads=8):
        self.ckpt, self.threads, self.handles = ckpt, threads, {}

    def handle(self, layer):
        if layer not in self.handles:
            from exllamav3.ext import exllamav3_ext as ext
            lists = []
            for w in ("w1", "w3", "w2"):
                for p in ("trellis", "suh", "svh"):
                    lists.append([self.ckpt.get(f"layers.{layer}.ffn.experts.{e}.{w}.{p}") for e in range(384)])
            self.handles[layer] = ext.exl3_moe_cpu_make_layer(*lists, [], [], [], 0, 10.0, 0)
        return self.handles[layer]

    def forward(self, layer, x_half, indices, weights):
        from exllamav3.ext import exllamav3_ext as ext
        out = torch.empty(x_half.shape[0], x_half.shape[1], dtype=torch.float32, device="cpu")
        ext.exl3_moe_cpu_forward(self.handle(layer), x_half.cpu(), indices.cpu().to(torch.int64),
                                 weights.cpu().to(torch.float16), out, self.threads)
        return out


STATE = {"store": None, "routes": RouteLog(), "act_fp8": True, "kernel": None,
         "moe_s": 0.0, "nll_targets": None, "nll": None, "cpu_experts": None}


def moe_forward(self, x, image_mask=None):
    """Replacement for model.MoE.forward with EXL3 routed experts; same math as Expert.forward."""
    t0 = time.perf_counter()
    shape = x.size()
    x = x.view(-1, self.dim)
    weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
    STATE["routes"].add(self.layer_id, indices, weights)
    y = torch.zeros_like(x, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
    active = [i for i, c in enumerate(counts) if c]
    kern = STATE["kernel"]
    xin = x.clone()
    if STATE["act_fp8"]:          # the official FP4 linear quantizes its input to FP8 first
        kern.act_quant(xin, 32, "ue8m0", torch.float8_e8m0fnu, True)
    xin = xin.half()
    lim = 10.0
    if STATE["cpu_experts"] is not None:
        y += STATE["cpu_experts"].forward(self.layer_id, xin, indices, weights).to(y.device)
        active = []
    for e, (w1, w3, w2) in STATE["store"].stream(self.layer_id, active):
        idx, top = torch.where(indices == e)
        xe = xin[idx].contiguous()
        gate = w1.forward(xe, {}, torch.float)
        up = w3.forward(xe, {}, torch.float)
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
        h = F.silu(gate) * up * weights[idx, top, None]
        h = h.to(x.dtype)
        if STATE["act_fp8"]:
            kern.act_quant(h, 32, "ue8m0", torch.float8_e8m0fnu, True)
        y[idx] += w2.forward(h.half().contiguous(), {}, torch.float)
    y += self.shared_experts(x)
    STATE["moe_s"] += time.perf_counter() - t0
    return y.type_as(x).view(shape)


# ---------------------------------------------------------------- engram rows from the SSD

class LazyEngramEmbedding(nn.Module):
    """Stand-in for ParallelEngramEmbedding: no 98 GB table; rows come from a file on lookup."""

    def __init__(self, num_embeddings, dim):
        super().__init__()
        self.num_embeddings, self.dim, self.block_size = num_embeddings, dim, 32
        self.file = None
        self.device = "cuda"
        self.pool = ThreadPoolExecutor(16)
        self.rows_read = 0

    def bind(self, st_file, layer_id):
        self.file = st_file
        _, wshape, self.w_off, _ = st_file.entries[f"layers.{layer_id}.engram.embed.weight"]
        _, sshape, self.s_off, _ = st_file.entries[f"layers.{layer_id}.engram.embed.scale"]
        assert wshape == [self.num_embeddings, self.dim] and sshape[0] == self.num_embeddings

    def _read_rows(self, rows, w_out, s_out):
        fd = self.file.fd
        for i, r in enumerate(rows):
            os.preadv(fd, [memoryview(w_out[i])], self.w_off + int(r) * self.dim)
            os.preadv(fd, [memoryview(s_out[i])], self.s_off + int(r) * (self.dim // self.block_size))

    def forward(self, indices):
        flat = indices.flatten().cpu().numpy()
        uniq, inv = np.unique(flat, return_inverse=True)
        w = np.empty((len(uniq), self.dim), dtype=np.uint8)
        s = np.empty((len(uniq), self.dim // self.block_size), dtype=np.uint8)
        parts = np.array_split(np.arange(len(uniq)), 16)
        list(self.pool.map(lambda p: self._read_rows(uniq[p], w[p[0]:p[-1] + 1] if len(p) else w[:0],
                                                     s[p[0]:p[-1] + 1] if len(p) else s[:0]), parts))
        self.rows_read += len(uniq)
        wt = torch.from_numpy(w).to(self.device).view(torch.float8_e4m3fn)
        st = torch.from_numpy(s).to(self.device).view(torch.float8_e8m0fnu)
        vals = wt.float().unflatten(-1, (-1, self.block_size)) * st.float().unsqueeze(-1)
        vals = vals.flatten(-2).to(torch.bfloat16)
        inv_t = torch.from_numpy(inv.reshape(flat.shape)).to(self.device)
        return vals[inv_t].view(*indices.shape, self.dim)


# ---------------------------------------------------------------- output head (bf16 storage)

class Bf16Head(nn.Module):
    """ParallelHead with bf16 weights. Logits are fp32 as in the reference; computed in vocab
    chunks so no fp32 copy of the whole matrix exists. With STATE['nll_targets'] set, it also
    returns the next-token NLL of every position (teacher forcing) in STATE['nll']."""

    def __init__(self, vocab_size, dim, norm_eps=1e-6, hc_eps=1e-6):
        super().__init__()
        self.vocab_size, self.dim = vocab_size, dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.bfloat16))

    def _logits(self, x):
        out = torch.empty(x.size(0), self.vocab_size, dtype=torch.float32, device=x.device)
        for v0 in range(0, self.vocab_size, 16384):
            v1 = min(self.vocab_size, v0 + 16384)
            out[:, v0:v1] = F.linear(x.float(), self.weight[v0:v1].float())
        return out

    def forward(self, x, full_logits=False):
        tgt = STATE["nll_targets"]
        if tgt is not None:                       # x: [1, T, d]; targets: [T-1]
            xs = x[0, :-1]
            nll = torch.empty(xs.size(0), dtype=torch.float32, device=x.device)
            for p0 in range(0, xs.size(0), 512):
                lg = self._logits(xs[p0:p0 + 512])
                nll[p0:p0 + 512] = torch.logsumexp(lg, -1) - lg.gather(1, tgt[p0:p0 + 512, None])[:, 0]
            STATE["nll"] = nll
        if not full_logits:
            x = x[:, -1]
        return self._logits(x.reshape(-1, self.dim)).view(*x.shape[:-1], self.vocab_size)


# ---------------------------------------------------------------- build and load

def build_model(model_dir, max_seq_len, kernels):
    kern, kname = select_kernels(kernels)
    import model as M
    STATE["kernel"] = kern
    orig_expert = M.Expert
    M.Expert = lambda dim, inter, dtype=None, swiglu_limit=0.0: (
        None if dtype == torch.float4_e2m1fn_x2 else orig_expert(dim, inter, dtype=dtype, swiglu_limit=swiglu_limit))
    M.ParallelEngramEmbedding = LazyEngramEmbedding
    M.ParallelHead = Bf16Head
    M.MoE.forward = moe_forward

    cfg = json.load(open(os.path.join(HERE, "ref", "config.json")))
    args = M.ModelArgs(**cfg)
    args.max_batch_size, args.max_seq_len = 1, max_seq_len
    args.temperature = 0.0
    args.dspark_block_size = 0
    args.vision_n_layers = 0
    args.expert_dtype = "fp4"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    torch.set_default_dtype(torch.bfloat16)
    t0 = time.perf_counter()
    with torch.device("cuda"):
        model = M.Transformer(args, tok)
    torch.set_default_device("cuda")
    ckpt = Checkpoint(model_dir)
    load_dense(model, ckpt)
    eng = SafetensorsFile
    for layer in model.layers:
        if layer.engram is not None:
            lid = layer.layer_id
            path = os.path.join(model_dir, "engrams", f"engram-layer-{lid:02d}.safetensors")
            layer.engram.embed.bind(eng(path), lid)
    STATE["store"] = ExpertStore(ckpt)
    STATE["ckpt"] = ckpt
    info = {"kernels": kname, "load_s": round(time.perf_counter() - t0, 1),
            "gpu_alloc_gib_after_load": round(torch.cuda.memory_allocated() / 2**30, 2)}
    print("model ready", info, flush=True)
    return model, tok, args, info


@torch.no_grad()
def load_dense(model, ckpt):
    missing = []
    for name, p in model.named_parameters():
        if name.endswith("attn.wo_a.weight"):
            w = ckpt.get(name).to("cuda").float()
            s = ckpt.get(name.replace("weight", "scale")).to("cuda").float()
            w = w.unflatten(0, (-1, 32)).unflatten(-1, (-1, 32)) * s[:, None, :, None]
            p.copy_(w.flatten(2, 3).flatten(0, 1).to(p.dtype))
        elif ckpt.has(name):
            p.copy_(ckpt.get(name).to("cuda").to(p.dtype))
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(f"{len(missing)} parameters not in checkpoint, e.g. {missing[:5]}")


# ---------------------------------------------------------------- runs

def prefill_doc(model, ids):
    """One teacher-forced pass over a whole document: routing + per-token NLL."""
    STATE["routes"].reset()
    STATE["moe_s"] = 0.0
    store = STATE["store"]
    b0, r0 = store.bytes_read, store.read_s
    x = torch.tensor([ids], device="cuda")
    STATE["nll_targets"] = x[0, 1:]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model(x, 0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    STATE["nll_targets"] = None
    nll = STATE["nll"].float().cpu().numpy()
    routes, weights = STATE["routes"].stacked()
    return {"seconds": round(dt, 2), "moe_s": round(STATE["moe_s"], 2),
            "expert_gb": round((store.bytes_read - b0) / 1e9, 2),
            "expert_read_thread_s": round(store.read_s - r0, 2),
            "mean_nll": float(nll.mean()), "ppl": float(np.exp(nll.mean()))}, routes, weights, nll


def generate(model, tok, ids, max_new, eos_id):
    STATE["routes"].reset()
    out, times = [], []
    x = torch.tensor([ids], device="cuda")
    t0 = time.perf_counter()
    nxt = model(x, 0)[0]
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    pos = len(ids)
    prefill_routes = STATE["routes"].stacked()
    STATE["routes"].reset()
    for _ in range(max_new):
        t = int(nxt.item())
        out.append(t)
        if t == eos_id:
            break
        t1 = time.perf_counter()
        nxt = model(torch.tensor([[t]], device="cuda"), pos)[0]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t1)
        pos += 1
    decode_routes = STATE["routes"].stacked() if times else (np.zeros((0, N_LAYERS, TOPK), np.int16),) * 2
    return out, prefill_s, times, prefill_routes, decode_routes


def chat_prompt(messages, thinking_mode):
    from encoding import encode_messages
    return encode_messages(messages, thinking_mode=thinking_mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus")
    ap.add_argument("--prompts")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--kernels", default="auto", choices=["auto", "tilelang", "torch"])
    ap.add_argument("--no-expert-act-fp8", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    STATE["act_fp8"] = not a.no_expert_act_fp8
    os.makedirs(os.path.join(a.out, "routes"), exist_ok=True)
    model, tok, args, info = build_model(a.model_dir, a.max_tokens + a.gen_tokens + 16, a.kernels)
    info.update({"expert_act_fp8": STATE["act_fp8"], "max_tokens": a.max_tokens})
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)

    if a.smoke:
        prompts = [{"id": "smoke_zh", "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
                    "thinking_mode": "chat"}]
        docs = []
        a.gen_tokens = min(a.gen_tokens, 24)
    else:
        prompts = [json.loads(l) for l in open(a.prompts)] if a.prompts else []
        docs = [json.loads(l) for l in open(a.corpus)] if a.corpus else []

    # Generations first: they are the quickest end-to-end proof that the model is right
    for p in prompts:
        text = chat_prompt(p["messages"], p.get("thinking_mode", "chat"))
        ids = tok.encode(text)
        out, pre_s, times, (pr, pw), (dr, dw) = generate(model, tok, ids, a.gen_tokens, tok.eos_token_id)
        rec = {"id": p["id"], "prompt_tokens": len(ids), "new_tokens": len(out), "prefill_s": round(pre_s, 2),
               "decode_s_per_token_median": round(float(np.median(times)), 3) if times else None,
               "output": tok.decode(out)}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with open(os.path.join(a.out, "generations.jsonl"), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        np.savez_compressed(os.path.join(a.out, "routes", f"gen_{p['id']}.npz"), prompt_ids=np.array(ids),
                            out_ids=np.array(out), prefill_routes=pr, prefill_weights=pw,
                            decode_routes=dr, decode_weights=dw)

    for d in docs:
        if "messages" in d:
            text = chat_prompt(d["messages"], d.get("thinking_mode", "chat"))
        else:
            text = d["text"]
        ids = tok.encode(text)[: a.max_tokens]
        if len(ids) < 64:
            continue
        rec, routes, weights, nll = prefill_doc(model, ids)
        rec.update({"id": d["id"], "kind": d["kind"], "tokens": len(ids)})
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with open(os.path.join(a.out, "docs.jsonl"), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        np.savez_compressed(os.path.join(a.out, "routes", f"doc_{d['id']}.npz"), ids=np.array(ids),
                            routes=routes, weights=weights, nll=nll.astype(np.float16))
    print("done", flush=True)


if __name__ == "__main__":
    main()
