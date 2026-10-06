#!/usr/bin/env python3
"""Host-only K13-02 indexing/numerical-model checks, not a GPU acceptance test.

Uses NumPy only for this optional developer test; the CUDA kernel adds no
runtime dependency. Run with OPENBLAS_NUM_THREADS=1 python check_staged.py.
WMMA's actual hardware accumulation order is not emulated or certified here.
"""
from pathlib import Path
import hashlib
import re
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "k13_sparse_attn_prefill.cu"
HEADS, DIM, MAX_IDX, CHUNK = 64, 512, 1024, 256


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def warp_sum(x):
    """The lane-zero result of the reference's shuffle-down reduction."""
    x = x.copy()
    for offset in (16, 8, 4, 2, 1):
        x[..., :offset] += x[..., offset:2 * offset]
    return x[..., 0]


def reference(q, rows, valid, sink, scale):
    n_idx = len(rows)
    # Reference: 32 lanes, 16 products per lane, then shuffle-down reduction.
    partial = np.zeros((HEADS, n_idx, 32), dtype=np.float32)
    for d in range(0, DIM, 32):
        partial += q[:, None, d:d + 32] * rows[None, :, d:d + 32]
    score = warp_sum(partial) * scale
    score[:, ~valid] = -np.inf
    maximum = np.maximum(score.max(axis=1, initial=-np.inf), np.float32(-1e30))
    exp = np.exp(score - maximum[:, None], dtype=np.float32)
    # ops.cu block_sum<256>: four positions/thread, shuffle-down within
    # each of eight warps, then serially sum the eight warp totals.
    padded = np.zeros((HEADS, MAX_IDX), dtype=np.float32)
    padded[:, :n_idx] = exp
    sums = np.zeros((HEADS, 256), dtype=np.float32)
    for first in range(0, MAX_IDX, 256):
        sums += padded[:, first:first + 256]
    warp_totals = warp_sum(sums.reshape(HEADS, 8, 32))
    denom = np.zeros(HEADS, dtype=np.float32)
    for warp in range(8):
        denom += warp_totals[:, warp]
    with np.errstate(over="ignore"):
        denom += np.exp(sink - maximum, dtype=np.float32)
    p = bf16(exp)
    numerator = np.zeros((HEADS, DIM), dtype=np.float32)
    for t in range(n_idx):
        if valid[t]:
            numerator += p[:, t, None] * rows[t, None, :]
    with np.errstate(invalid="ignore"):
        return bf16(numerator / denom[:, None])


def staged_model(q, rows, valid, sink, scale):
    n_idx = len(rows)
    padded_n = ((n_idx + 63) // 64) * 64
    keys = np.zeros((padded_n, DIM), dtype=np.float32)
    keys[:n_idx] = rows
    score = np.zeros((HEADS, padded_n), dtype=np.float32)
    # Mathematical 16x16x16 tile model. These are CPU matrix products,
    # not a claim of bitwise reproduction of BF16 tensor-core instructions.
    for first in range(0, padded_n, 64):
        for d in range(0, DIM, 16):
            score[:, first:first + 64] += q[:, d:d + 16] @ keys[first:first + 64, d:d + 16].T
    score *= scale
    full = np.full((HEADS, MAX_IDX), -np.inf, dtype=np.float32)
    full[:, :n_idx] = score[:, :n_idx]
    full[:, :n_idx][:, ~valid] = -np.inf
    maximum = np.maximum(full.max(axis=1), np.float32(-1e30))
    p = np.exp(full - maximum[:, None], dtype=np.float32)
    lane_sums = np.zeros((HEADS, 32), dtype=np.float32)
    for first in range(0, MAX_IDX, 32):
        lane_sums += p[:, first:first + 32]
    # XOR reduction has the same lane-zero sum tree as shuffle-down.
    denom = warp_sum(lane_sums)
    with np.errstate(over="ignore"):
        denom += np.exp(sink - maximum, dtype=np.float32)
    p = bf16(p)
    numerator = np.zeros((HEADS, DIM), dtype=np.float32)
    pv_n = ((n_idx + 31) // 32) * 32
    values = np.zeros((pv_n, DIM), dtype=np.float32)
    values[:n_idx] = rows
    for first in range(0, pv_n, 16):
        for d in range(0, DIM, 64):
            numerator[:, d:d + 64] += p[:, first:first + 16] @ values[first:first + 16, d:d + 64]
    with np.errstate(invalid="ignore"):
        return bf16(numerator / denom[:, None])


def indexing_checks():
    # Every head/key and head/dimension is owned by exactly one warp fragment.
    qk = np.zeros((HEADS, 64), dtype=np.uint8)
    pv = np.zeros((HEADS, DIM), dtype=np.uint8)
    for warp in range(8):
        head = (warp // 4) * 32
        for h in (0, 16):
            qk[head + h:head + h + 16, (warp % 4) * 16:(warp % 4 + 1) * 16] += 1
            for first_dim in range(0, DIM, 64):
                dim = first_dim + (warp % 4) * 16
                pv[head + h:head + h + 16, dim:dim + 16] += 1
    assert np.all(qk == 1) and np.all(pv == 1)
    # Full public query bound, including every possible final chunk size.
    for m in range(1, 16385):
        chunks = [(first, min(CHUNK, m - first)) for first in range(0, m, CHUNK)]
        assert sum(count for _, count in chunks) == m
        assert chunks[-1][0] + chunks[-1][1] == m
        assert max(count for _, count in chunks) <= CHUNK
    # Every index-list length, including zero, has a safe padded scratch range.
    for n in range(MAX_IDX + 1):
        assert ((n + 63) // 64) * 64 <= MAX_IDX
        assert ((n + 31) // 32) * 32 <= MAX_IDX
        for first in range(0, n, 32):
            assert first + 31 < MAX_IDX
    # WMMA base pointers are 32-byte aligned and vector moves 16-byte aligned.
    for stride, rows in ((72, 64), (40, 64), (72, 32)):
        for r in range(rows):
            assert (r * stride * 2) % 16 == 0
        for r in range(0, rows, 16):
            for col in range(0, stride - 8, 16):
                assert (2 * (r * stride + col)) % 32 == 0
    elements = CHUNK * HEADS * MAX_IDX
    assert elements * 6 + CHUNK * HEADS * 4 == 100728832
    # The interface's int32 row IDs require 64-bit address arithmetic.
    assert ((2**31 - 1) * DIM * 2) > 2**32
    source = SOURCE.read_text()
    expected = {"kHeads": HEADS, "kDim": DIM, "kMaxIndices": MAX_IDX,
                "kQueryChunk": CHUNK, "kQKRows": 64, "kQKDepth": 64,
                "kPVDim": 64, "kPVDepth": 32}
    for name, value in expected.items():
        assert re.search(rf"constexpr int {name} = {value};", source), name
    assert source.count("cudaMalloc(") == 1
    assert "if (found != scratch.end()) return found->second;" in source
    assert source.count("static_cast<size_t>(j) * kDim") == 2
    for forbidden in ("cudaFree(", "cudaMallocAsync", "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpy"):
        assert forbidden not in source
    print("PASS exhaustive chunk/index bounds, output ownership, alignment, source invariants")


def main():
    indexing_checks()
    rng = np.random.default_rng(1302)
    worst = 0.0
    for n_idx in (0, 1, 15, 31, 32, 33, 63, 64, 65, 127, 128, 129, 639, 640, 641, 1023, 1024):
        q = bf16(rng.uniform(-1, 1, (HEADS, DIM)))
        kv = bf16(rng.uniform(-1, 1, (73, DIM)))
        idx = rng.integers(-1, len(kv), n_idx)
        # Arbitrary, duplicate, unsorted indices and both scattered/all padding.
        if n_idx in (1, 33):
            idx[:] = -1
        elif n_idx:
            idx[::7] = -1
        valid = idx >= 0
        rows = kv[np.maximum(idx, 0)].copy()
        rows[~valid] = 0
        sink = rng.uniform(-5, 5, HEADS).astype(np.float32)
        scale = np.float32((-1 if n_idx == 641 else 1) / np.sqrt(DIM))
        expected = reference(q, rows, valid, sink, scale)
        actual = staged_model(q, rows, valid, sink, scale)
        error = float(np.linalg.norm(actual.astype(np.float64) - expected) /
                      max(float(np.linalg.norm(expected.astype(np.float64))), 1e-30))
        assert np.isfinite(actual).all(), n_idx
        assert error <= 3e-3, (n_idx, error)
        worst = max(worst, error)
        print(f"PASS CPU-model n_idx={n_idx:4d} rel_l2={error:.8g}")
    print(f"PASS 17 CPU numerical-model cases; worst_rel_l2={worst:.8g}")
    print("source_sha256=" + hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    print("GPU numerical parity, graph capture/replay, sanitizer and speed: NOT RUN")


if __name__ == "__main__":
    main()
