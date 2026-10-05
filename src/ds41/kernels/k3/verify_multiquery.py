#!/usr/bin/env python3
"""CPU model for K3-07's paired index masking and online BF16-P numerics.

This optional development check uses the repository's existing NumPy dependency.
It does not replace the fixed GPU acceptance test or execute CUDA code.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def scores(q, kv, parts):
    # Match the CTA's FP32 split of the 512-wide contraction. NumPy does not
    # simulate the internal tensor-core FP32 reduction order.
    chunks = [q[:, d:d + 512 // parts] @ kv[:, d:d + 512 // parts].T
              for d in range(0, 512, 512 // parts)]
    return chunks[0] + chunks[1] if parts == 2 else (chunks[0] + chunks[1]) + (chunks[2] + chunks[3])


def model(q, kv, index, sink):
    m, heads, _ = q.shape
    paired = m > 1
    group = 2 if paired else 1
    step = 16 if paired else 32
    output = np.zeros_like(q)
    common_count = 0
    for first_query in range(0, m, group):
        n = min(group, m - first_query)
        query = np.zeros((group * heads, 512), np.float32)
        query[:n * heads] = q[first_query:first_query + n].reshape(n * heads, 512)
        maximum = np.full(group * heads, -1e30, np.float32)
        denom = np.zeros(group * heads, np.float32)
        numerator = np.zeros((group * heads, 512), np.float32)
        for first in range(0, index.shape[1], step):
            ids = np.full((group, step), -1, np.int32)
            length = min(step, index.shape[1] - first)
            ids[:n, :length] = index[first_query:first_query + n, first:first + length]
            common = paired and np.array_equal(ids[0], ids[1]) and np.all(ids < 128)
            common_count += common
            flat = ids[0] if common else ids.reshape(-1)
            valid_rows = flat >= 0
            rows = np.zeros((len(flat), 512), np.float32)
            rows[valid_rows] = kv[flat[valid_rows]]
            s = scores(query, rows, 4 if common else 2) * np.float32(1 / np.sqrt(512))
            for h in range(group * heads):
                owned = np.ones(len(flat), bool) if common or not paired else np.arange(len(flat)) // step == h // heads
                s[h, ~(owned & valid_rows)] = -np.inf
            next_max = np.maximum(maximum, s.max(axis=1))
            alpha = np.exp(maximum - next_max)
            p = np.exp(s - next_max[:, None])
            denom = denom * alpha + p.sum(axis=1)
            numerator = numerator * alpha[:, None] + bf16(p) @ rows
            maximum = next_max
        with np.errstate(over="ignore"):
            denom += np.exp(np.tile(sink, group) - maximum)
        output[first_query:first_query + n] = bf16(numerator[:n * heads] / denom[:n * heads, None]).reshape(n, heads, 512)
    return output, common_count


def reference(q, kv, index, sink):
    out = np.empty_like(q)
    for t in range(len(q)):
        ids = index[t][index[t] >= 0]
        rows = kv[ids]
        s = q[t] @ rows.T * np.float32(1 / np.sqrt(512))
        maximum = np.maximum(-1e30, s.max(axis=1)) if len(ids) else np.full(q.shape[1], -1e30, np.float32)
        p = np.exp(s - maximum[:, None])
        with np.errstate(over="ignore"):
            denom = p.sum(axis=1) + np.exp(sink - maximum)
        out[t] = bf16((bf16(p) @ rows) / denom[:, None])
    return out


def validate_mma_layout():
    # A = key[16,16]; B = q[8,16]^T. x4 ldmatrix's four
    # quadrants and the x2 B load cover the documented m16n8k16 maps.
    a = np.arange(256).reshape(16, 16)
    q = 1000 + np.arange(128).reshape(8, 16)
    for lane in range(32):
        group, thread = divmod(lane, 4)
        for reg in range(4):
            row = group + (8 if reg & 1 else 0)
            col = thread * 2 + (8 if reg & 2 else 0)
            assert a[row, col + 1] - a[row, col] == 1
        # B's N columns are the head dimension, exactly the output lanes.
        for reg in range(2):
            assert q[group, thread * 2 + reg * 8] == 1000 + group * 16 + thread * 2 + reg * 8
    # Transposed PV load: x4's supplied addresses enumerate (key, dim)
    # quadrants 00, 01, 10, 11. MMA consumes (dim, key) quadrants
    # 00, 10, 01, 11, which is exactly each 8x8 transpose's interpretation.
    v = np.arange(256).reshape(16, 16)
    rebuilt = np.empty_like(v)
    for quadrant in range(4):
        key0, dim0 = (quadrant // 2) * 8, (quadrant % 2) * 8
        rebuilt[dim0:dim0 + 8, key0:key0 + 8] = v[key0:key0 + 8, dim0:dim0 + 8].T
    assert np.array_equal(rebuilt, v.T)
    # Four warps cover every output exactly once, with even/odd heads
    # staying within one token. The single-query path discards padded heads.
    for paired, output_dim in [(False, 64), (True, 128)]:
        seen = set()
        for warp in range(4):
            for lane in range(32):
                h0, r = (lane % 4) * 2, lane // 4
                for h in [h0, h0 + 1]:
                    if h >= (8 if paired else 4):
                        continue
                    for tile in range(output_dim // 64):
                        for dr in [0, 8]:
                            d = (warp * (output_dim // 64) + tile) * 16 + r + dr
                            assert (h, d) not in seen
                            seen.add((h, d))
        assert len(seen) == (8 if paired else 4) * output_dim


def main():
    validate_mma_layout()
    rng = np.random.default_rng(7007)
    kv = bf16(rng.normal(size=(1152, 512)))
    sink = rng.normal(size=4).astype(np.float32)
    cases = [(1, 640), (4, 640), (8, 640), (1, 128), (2, 300), (3, 1024)]
    cases += [(m, size) for m, size in [(1, 0), (2, 1), (3, 15), (4, 16), (5, 17), (6, 31), (7, 32), (8, 33)]]
    worst, common_total = 0., 0
    for m, size in cases:
        for pattern in ["common_window", "different_lists", "negative_holes", "all_empty"]:
            q = bf16(rng.normal(size=(m, 4, 512)))
            index = rng.integers(128, len(kv), size=(m, size), dtype=np.int32)
            if pattern == "common_window":
                count = min(128, size)
                # Shuffled common windows must still work; list position is
                # not used as the KV row. Duplicate compressed IDs are legal.
                index[:, :count] = rng.permutation(128)[:count]
            elif pattern == "different_lists":
                index = rng.integers(0, len(kv), size=(m, size), dtype=np.int32)
            elif pattern == "negative_holes":
                index = rng.integers(0, len(kv), size=(m, size), dtype=np.int32)
                index[rng.random(index.shape) < .43] = -rng.integers(1, 9)
            else:
                index.fill(-1)
            got, common = model(q, kv, index, sink)
            ref = reference(q, kv, index, sink)
            err = float(np.linalg.norm(got - ref) / max(float(np.linalg.norm(ref)), 1e-30))
            assert np.isfinite(got).all() and err <= .003, (m, size, pattern, err)
            worst = max(worst, err)
            common_total += common
            print(f"m={m} n_idx={size} pattern={pattern} rel_l2={err:.6g} common_tiles={common}")
    assert common_total > 0
    print(f"PASS: 56 CPU cases, MMA/output mapping; worst_rel_l2={worst:.6g}")
    print("GPU correctness, launch behavior, graph capture, and timings remain untested.")


if __name__ == "__main__":
    main()
