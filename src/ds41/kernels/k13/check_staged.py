#!/usr/bin/env python3
"""Host-only K13-04 indexing/numerical-model checks, not a GPU acceptance test.

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
QK_ROWS, QK_THREADS, QK_DEPTH = 128, 512, 64
QK_KEY_WARPS = QK_ROWS // 16
PV_THREADS, PV_DIM, PV_DEPTH = 256, 64, 32


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
    padded_n = ((n_idx + QK_ROWS - 1) // QK_ROWS) * QK_ROWS
    keys = np.zeros((padded_n, DIM), dtype=np.float32)
    keys[:n_idx] = rows
    score = np.zeros((HEADS, padded_n), dtype=np.float32)
    # Mathematical 16x16x16 tile model. These are CPU matrix products,
    # not a claim of bitwise reproduction of BF16 tensor-core instructions.
    for first in range(0, padded_n, QK_ROWS):
        for d in range(0, DIM, 16):
            score[:, first:first + QK_ROWS] += q[:, d:d + 16] @ keys[first:first + QK_ROWS, d:d + 16].T
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


def source_checks(source):
    """Pin the CPU model to the candidate's actual launch and indexing scheme.

    These lexical guards supplement the ownership model and CUDA build. They do
    not prove arbitrary CUDA source is safe or replace a GPU execution test.
    """
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"\s+", " ", source)
    expected = {"kHeads": HEADS, "kDim": DIM, "kMaxIndices": MAX_IDX,
                "kQueryChunk": CHUNK, "kThreads": PV_THREADS,
                "kQKRows": QK_ROWS, "kQKThreads": QK_THREADS,
                "kQKDepth": QK_DEPTH, "kPVDim": PV_DIM, "kPVDepth": PV_DEPTH}
    for name, value in expected.items():
        assert f"constexpr int {name} = {value};" in source, name
    required = (
        "constexpr int kWarps = kThreads / 32;",
        "constexpr int kQKKeyWarps = kQKRows / 16;",
        "constexpr int kQKStride = kQKDepth + 8;",
        "constexpr int kPStride = kPVDepth + 8;",
        "constexpr int kVStride = kPVDim + 8;",
        'static_assert(sizeof(QKStorage) == 28160, "QK shared layout changed");',
        'static_assert(sizeof(PVStorage) == 9728, "PV shared layout changed");',
        "__launch_bounds__(kQKThreads, 2) void qk_stage(",
        "if (found != scratch.end()) return found->second;",
        "if (m <= 0) return;",
        "for (int first = 0; first < m; first += kQueryChunk)",
        "const int count = m - first < kQueryChunk ? m - first : kQueryChunk;",
        "const size_t query_offset = static_cast<size_t>(first) * kHeads * kDim;",
        "const int32_t* indices = idx + static_cast<size_t>(first) * n_idx;",
        "if (n_idx > 0) qk_stage<<<dim3((n_idx + kQKRows - 1) / kQKRows, count), kQKThreads, 0, stream>>>",
        "softmax_stage<<<dim3(kHeads / kWarps, count), kThreads, 0, stream>>>",
        "pv_stage<<<dim3(kDim / kPVDim, count), kThreads, 0, stream>>>",
        "const bool valid = t < n_idx && idx[static_cast<size_t>(query) * n_idx + t] >= 0;",
        "float maximum = -1.0e30f;",
        "sum += p; probabilities[row * kMaxIndices + lane + i * 32] = __float2bfloat16_rn(p);",
        "if (lane == 0) denominators[row] = sum + expf(sink[head] - maximum);",
    )
    for fragment in required:
        assert fragment in source, fragment
    qk = source.split("void qk_stage(", 1)[1].split("void softmax_stage(", 1)[0]
    for fragment in (
        "const int head = (warp / kQKKeyWarps) * 32;",
        "const int row = (warp % kQKKeyWarps) * 16;",
        "if (threadIdx.x < kQKRows)",
        "shared.indices[threadIdx.x] = t < n_idx ? idx[static_cast<size_t>(query) * n_idx + t] : -1;",
        "i < kHeads * kQKDepth / 8; i += kQKThreads",
        "i < kQKRows * kQKDepth / 8; i += kQKThreads",
        "wmma::load_matrix_sync(b, shared.k + row * kQKStride + d, kQKStride);",
        "wmma::load_matrix_sync(a, shared.q + (head + h * 16) * kQKStride + d, kQKStride);",
        "scores + (static_cast<size_t>(query) * kHeads + head + h * 16) * kMaxIndices + first + row;",
        "wmma::store_matrix_sync(dst, accum[h], kMaxIndices, wmma::mem_row_major);",
    ):
        assert fragment in qk, fragment
    assert qk.count("__syncthreads();") == 3
    assert qk.count("if (j >= 0)") == 1
    assert source.count("cudaMalloc(") == 1
    assert source.count("static_cast<size_t>(j) * kDim") == 2
    assert source.count("0, stream>>>") == 3
    for forbidden in ("cudaFree", "cudaMallocAsync", "cudaMallocFromPoolAsync",
                      "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpy"):
        assert forbidden not in source, forbidden


def negative_source_checks(source):
    """Prove the guards reject representative regressions, without editing CUDA."""
    mutations = (
        ("stale QK width", "constexpr int kQKRows = 128;", "constexpr int kQKRows = 64;"),
        ("stale QK thread count", "constexpr int kQKThreads = 512;", "constexpr int kQKThreads = 256;"),
        ("wrong head ownership", "(warp / kQKKeyWarps) * 32", "(warp / 4) * 32"),
        ("wrong key ownership", "(warp % kQKKeyWarps) * 16", "(warp % 4) * 16"),
        ("overlapping Q loads", "i < kHeads * kQKDepth / 8; i += kQKThreads", "i < kHeads * kQKDepth / 8; i += kThreads"),
        ("overlapping K loads", "i < kQKRows * kQKDepth / 8; i += kQKThreads", "i < kQKRows * kQKDepth / 8; i += kThreads"),
        ("old QK launch", "count), kQKThreads, 0, stream>>>", "count), kThreads, 0, stream>>>"),
        ("unaligned QK stride", "kQKStride = kQKDepth + 8", "kQKStride = kQKDepth + 7"),
        ("missing tile barrier", "__syncthreads();", "/* barrier removed */"),
        ("unguarded padded index", "t < n_idx ? idx[", "true ? idx["),
        ("overflowing row address", "static_cast<size_t>(j) * kDim", "j * kDim"),
        ("masked row read", "t < n_idx && idx[", "t <= n_idx && idx["),
        ("default stream", "0, stream>>>", "0, nullptr>>>"),
        ("scratch reallocation", "if (found != scratch.end()) return found->second;", "/* retained scratch ignored */"),
        ("host synchronization", "if (m <= 0) return;", "if (m <= 0) return; cudaDeviceSynchronize();"),
        ("rounded denominator", "sum += p;", "sum += __bfloat162float(__float2bfloat16_rn(p));"),
        ("sink omitted", "sum + expf(sink[head] - maximum)", "sum"),
    )
    for label, old, new in mutations:
        assert old in source, label
        try:
            source_checks(source.replace(old, new, 1))
        except AssertionError:
            continue
        raise AssertionError(f"negative guard accepted {label}")
    print(f"PASS {len(mutations)} negative source mutations rejected")


def vector_coverage(rows, depth, threads):
    coverage = np.zeros((rows, depth), dtype=np.uint8)
    for tid in range(threads):
        for i in range(tid, rows * depth // 8, threads):
            row, vector = divmod(i, depth // 8)
            coverage[row, vector * 8:vector * 8 + 8] += 1
    assert np.all(coverage == 1), (rows, depth, threads)


def indexing_checks():
    # QK uses sixteen warps; the unchanged PV stage still uses eight.
    qk = np.zeros((HEADS, QK_ROWS), dtype=np.uint8)
    for warp in range(QK_THREADS // 32):
        head = (warp // QK_KEY_WARPS) * 32
        row = (warp % QK_KEY_WARPS) * 16
        for h in (0, 16):
            qk[head + h:head + h + 16, row:row + 16] += 1
    assert np.all(qk == 1)
    pv = np.zeros((HEADS, DIM), dtype=np.uint8)
    for warp in range(PV_THREADS // 32):
        head = (warp // 4) * 32
        for h in (0, 16):
            for first_dim in range(0, DIM, PV_DIM):
                dim = first_dim + (warp % 4) * 16
                pv[head + h:head + h + 16, dim:dim + 16] += 1
    assert np.all(pv == 1)
    # Each operand value is published exactly once by cooperative vector loads.
    for rows, depth, threads in ((HEADS, QK_DEPTH, QK_THREADS),
                                  (QK_ROWS, QK_DEPTH, QK_THREADS),
                                  (HEADS, PV_DEPTH, PV_THREADS),
                                  (PV_DEPTH, PV_DIM, PV_THREADS)):
        vector_coverage(rows, depth, threads)
    # Full public query bound, including every possible final chunk size.
    for m in range(1, 16385):
        chunks = [(first, min(CHUNK, m - first)) for first in range(0, m, CHUNK)]
        assert sum(count for _, count in chunks) == m
        assert chunks[-1][0] + chunks[-1][1] == m
        assert max(count for _, count in chunks) <= CHUNK
    # Every index-list length, including zero, has a safe padded scratch range.
    for n in range(MAX_IDX + 1):
        qk_end = ((n + QK_ROWS - 1) // QK_ROWS) * QK_ROWS
        assert qk_end <= MAX_IDX
        assert ((n + PV_DEPTH - 1) // PV_DEPTH) * PV_DEPTH <= MAX_IDX
        stores = np.zeros(MAX_IDX, dtype=np.uint8)
        for first in range(0, n, QK_ROWS):
            stores[first:first + QK_ROWS] += 1
        assert np.all(stores[:qk_end] == 1) and np.all(stores[qk_end:] == 0)
        for first in range(0, n, PV_DEPTH):
            assert first + PV_DEPTH - 1 < MAX_IDX
    # WMMA base pointers are 32-byte aligned and vector moves 16-byte aligned.
    for stride, rows in ((72, HEADS), (72, QK_ROWS), (40, HEADS), (72, PV_DEPTH)):
        for r in range(rows):
            assert (r * stride * 2) % 16 == 0
        for r in range(0, rows, 16):
            for col in range(0, stride - 8, 16):
                assert (2 * (r * stride + col)) % 32 == 0
    for head in range(HEADS):
        for first in range(0, MAX_IDX, 16):
            assert ((head * MAX_IDX + first) * 4) % 32 == 0
    elements = CHUNK * HEADS * MAX_IDX
    assert elements * 6 + CHUNK * HEADS * 4 == 100728832
    assert (HEADS + QK_ROWS) * (QK_DEPTH + 8) * 2 + QK_ROWS * 4 == 28160
    # The interface's int32 row IDs require 64-bit address arithmetic.
    assert ((2**31 - 1) * DIM * 2) > 2**32
    source = SOURCE.read_text()
    source_checks(source)
    negative_source_checks(source)
    print("PASS exhaustive chunk/index bounds, 128-row QK and PV ownership, vector loads, alignment, source invariants")


def main():
    indexing_checks()
    rng = np.random.default_rng(1302)
    worst = 0.0
    for n_idx in (0, 1, 15, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 639, 640, 641, 1023, 1024):
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
    print(f"PASS 20 CPU numerical-model cases; worst_rel_l2={worst:.8g}")
    print("source_sha256=" + hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    print("GPU numerical parity, graph capture/replay, sanitizer and speed: NOT RUN")


if __name__ == "__main__":
    main()
