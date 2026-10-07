"""tools/ds41/make_profile.py - the DeepSeek V4.1 Flash expert-cache profile: every (layer, expert) pair, ranked.

The engine fills its VRAM expert slots with the profile's pairs in order, so the ranking decides which experts
start resident. The file format is upstream's tools/make_profile.py format, with 40 layers x 384 experts:
`STRP`, then uint32 `1, n_layers, n_expert, n_ranked, n_ranked`, then the pairs (uint16 layer, uint16 expert),
then the n_layers x n_expert int32 table of each pair's rank.

The order: the pairs the routing traces used, most frequent first (ties: lower layer, then lower expert), then every
pair still missing, interleaved across the layers.

    python tools/ds41/make_profile.py TRACE.npz ... --out ds41/data/expert-profile.bin

A trace is a ds41/proto routing record: `routes` [T, 40, 6] (doc_*.npz) or `decode_routes` (gen_*.npz).
"""
import argparse
import struct
from collections import Counter

import numpy as np

N_LAYER, N_EXPERT, TOP_K = 40, 384, 6
MAGIC, VERSION = b"STRP", 1


def trace_routes(path):
    z = np.load(path)
    for key in ("routes", "decode_routes"):
        if key in z.files and len(z[key]):
            r = np.asarray(z[key], dtype=np.int64)
            if r.ndim != 3 or r.shape[1:] != (N_LAYER, TOP_K):
                raise SystemExit(f"{path}: {key} has shape {r.shape}, not [T, {N_LAYER}, {TOP_K}]")
            return r
    return np.zeros((0, N_LAYER, TOP_K), np.int64)


def count_pairs(routes):
    """routes [T, L, K] -> Counter of (layer, expert)."""
    c = Counter()
    flat = (np.arange(N_LAYER)[None, :, None] * N_EXPERT + routes).reshape(-1)
    for idx, n in zip(*np.unique(flat, return_counts=True)):
        c[(int(idx) // N_EXPERT, int(idx) % N_EXPERT)] += int(n)
    return c


def rank(freq):
    ranked = [p for p, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    seen = set(ranked)
    ranked += [(l, e) for e in range(N_EXPERT) for l in range(N_LAYER) if (l, e) not in seen]
    return ranked


def write_profile(path, ranked):
    table = np.full((N_LAYER, N_EXPERT), -1, np.int32)
    for slot, (l, e) in enumerate(ranked):
        table[l, e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, N_LAYER, N_EXPERT, len(ranked), len(ranked)))
        f.write(np.asarray(ranked, np.uint16).tobytes())
        f.write(table.astype("<i4").tobytes())


def read_profile(path):
    blob = open(path, "rb").read()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not a Strata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if (ver, nl, ne) != (VERSION, N_LAYER, N_EXPERT):
        raise SystemExit(f"{path}: version {ver}, {nl}x{ne}, not {VERSION}, {N_LAYER}x{N_EXPERT}")
    pairs = np.frombuffer(blob, np.uint16, 2 * n, 24).reshape(n, 2)
    return [(int(l), int(e)) for l, e in pairs]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    freq = Counter()
    tokens = 0
    for t in a.traces:
        r = trace_routes(t)
        tokens += len(r)
        freq += count_pairs(r)
    ranked = rank(freq)
    write_profile(a.out, ranked)
    assert read_profile(a.out) == ranked, "the profile did not survive the round trip"
    print(f"wrote {a.out}: {len(ranked)} ranked pairs ({len(freq)} from {tokens} traced tokens, "
          f"{len(ranked) - len(freq)} filled in)")


if __name__ == "__main__":
    main()
