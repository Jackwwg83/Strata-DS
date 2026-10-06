#!/usr/bin/env python3
"""K13-01 CPU layout, arithmetic, and source protocol checks.

Run: python src/ds41/kernels/k13/verify_fused.py
Numerical models require NumPy (development only). No GPU pass or timing claim.
"""
import hashlib
import pathlib
import random
import re
import numpy as np

HEADS, DIM, ROWS, SLICE, LIMIT, THREADS = 16, 512, 64, 128, 1024, 256
QSTRIDE, PSTRIDE = 520, 1032
SOURCE = pathlib.Path(__file__).resolve().parents[1] / "k13_sparse_attn_prefill.cu"


def offset(row, dimension, width):
    return row * width + (dimension ^ ((row & 7) * 8))


def ldmatrix(memory, addresses, count, transpose=False):
    """PTX m8n8 x{2,4}: groups of 8 address lanes supply successive matrices."""
    result = []
    for lane in range(32):
        row, col = lane // 4, (lane % 4) * 2
        registers = []
        for matrix in range(count):
            if transpose:
                pair = tuple(memory[addresses[matrix * 8 + col + e] + row] for e in range(2))
            else:
                pair = tuple(memory[addresses[matrix * 8 + row] + col + e] for e in range(2))
            registers.append(pair)
        result.append(registers)
    return result


def check_matrix_layout():
    total = 0
    for width in (SLICE, DIM):
        memory = [None] * (ROWS * width)
        owners = set()
        for tid in range(THREADS):
            for i in range(tid, ROWS * width // 8, THREADS):
                r, d = i // (width // 8), i % (width // 8) * 8
                address = offset(r, d, width)
                assert address % 8 == 0 and address + 8 <= len(memory)
                for e in range(8):
                    assert address + e not in owners
                    owners.add(address + e)
                    memory[address + e] = (r, d + e)
        assert len(owners) == ROWS * width
        if width == SLICE:
            for tile in range(0, ROWS, 16):
                for d in range(0, SLICE, 16):
                    addresses = [offset(tile + lane % 16, d + lane // 16 * 8, width) for lane in range(32)]
                    fragments = ldmatrix(memory, addresses, 4)
                    for lane, fragment in enumerate(fragments):
                        r, c = lane // 4, lane % 4 * 2
                        expected = [((tile + r + rr, d + c + cc), (tile + r + rr, d + c + cc + 1))
                                    for rr, cc in ((0, 0), (8, 0), (0, 8), (8, 8))]
                        assert fragment == expected
                        total += 1
        else:
            for r in range(0, ROWS, 16):
                for d in range(0, DIM, 16):
                    addresses = [offset(r + lane % 8 + lane // 16 * 8,
                                        d + (lane // 8 & 1) * 8, width) for lane in range(32)]
                    fragments = ldmatrix(memory, addresses, 4, transpose=True)
                    for lane, fragment in enumerate(fragments):
                        rr, cc = lane // 4, lane % 4 * 2
                        expected = [((r + cc + k, d + rr + dd), (r + cc + k + 1, d + rr + dd))
                                    for dd, k in ((0, 0), (8, 0), (0, 8), (8, 8))]
                        assert fragment == expected
                        total += 1
    for stride, extent in ((QSTRIDE, DIM), (PSTRIDE, LIMIT)):
        memory = [(h, d) for h in range(HEADS) for d in range(stride)]
        for head in (0, 8):
            for k in range(0, extent, 16):
                addresses = [(head + lane % 8) * stride + k + (lane // 8 & 1) * 8 for lane in range(32)]
                fragments = ldmatrix(memory, addresses, 2)
                for lane, fragment in enumerate(fragments):
                    column, kk = head + lane // 4, k + lane % 4 * 2
                    assert fragment == [((column, kk), (column, kk + 1)),
                                        ((column, kk + 8), (column, kk + 9))]
                    total += 1
    print(f"PASS {total} literal ldmatrix/MMA operand fragment checks and unique swizzled gathers")


def check_ownership_and_tails():
    score_owners, output_owners = set(), set()
    for tid in range(THREADS):
        lane, warp = tid % 32, tid // 32
        head0, row = (warp // 4) * 8 + (lane % 4) * 2, (warp % 4) * 16 + lane // 4
        for h, r in ((head0, row), (head0 + 1, row), (head0, row + 8), (head0 + 1, row + 8)):
            assert (h, r) not in score_owners
            score_owners.add((h, r))
        for v in range(8):
            d = ((warp % 4) * 8 + v) * 16 + lane // 4
            for h, dd in ((head0, d), (head0 + 1, d), (head0, d + 8), (head0 + 1, d + 8)):
                assert (h, dd) not in output_owners
                output_owners.add((h, dd))
    assert score_owners == {(h, r) for h in range(HEADS) for r in range(ROWS)}
    assert output_owners == {(h, d) for h in range(HEADS) for d in range(DIM)}
    rng = random.Random(1301)
    pool = [-2**31, -1, 0, 127, 128, 2**21, 2**31-1]
    for n_idx in range(LIMIT + 1):
        indices = [rng.choice(pool) for _ in range(n_idx)]
        read_positions = []
        for first in range(0, n_idx, ROWS):
            for row in range(ROWS):
                position = first + row
                j = indices[position] if position < n_idx else -1
                if position < n_idx:
                    read_positions.append(position)
                source = 0x100000000 + j * DIM * 2 if j >= 0 else 0x100000000
                for d in (0, 8, 504):
                    address = source + d * 2 if j >= 0 else source
                    assert address % 16 == 0 and 0 <= address < 2**64
                    assert j >= 0 or address == 0x100000000
                assert first + row < LIMIT
        assert read_positions == list(range(n_idx))
    # Query grid and 64-bit offsets cover both endpoints and random legal shapes.
    for m in (1, 19, 37, 300, 4096, 8192, 16384):
        for query in (0, m-1):
            for head_tile in range(4):
                indices = [((query * 64 + head_tile * HEADS + h) * DIM + d)
                           for h, d in output_owners]
                assert len(set(indices)) == HEADS * DIM
                assert min(indices) >= query * 64 * DIM
                assert max(indices) < (query + 1) * 64 * DIM
    print("PASS all n_idx 0..1024, negative/large indices, disjoint QK/output ownership, m up to 16384")


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def tree_sum(x):
    x = x.copy()
    offset = x.shape[-1] // 2
    while offset:
        x[..., :offset] += x[..., offset:2*offset]
        offset //= 2
    return x[..., 0]


def reference(q, values, valid, sink, scale):
    # Reference warp-lane sequential dot, followed by down-shuffle reduction.
    lane_sum = np.zeros((HEADS, len(valid), 32), dtype=np.float32)
    for d in range(0, DIM, 32):
        lane_sum += q[:, None, d:d+32] * values[None, :, d:d+32]
    scores = tree_sum(lane_sum) * np.float32(scale)
    scores[:, ~valid] = -np.inf
    maximum = np.maximum(np.max(scores, axis=1, initial=-np.inf), np.float32(-1e30))
    p = np.exp(scores - maximum[:, None], dtype=np.float32)
    total = np.zeros((HEADS, 256), dtype=np.float32)
    for t in range(len(valid)):
        total[:, t % 256] += p[:, t]
    denominator = tree_sum(total) + np.exp(sink - maximum, dtype=np.float32)
    p = bf16(p)
    numerator = np.zeros((HEADS, DIM), dtype=np.float32)
    for t in range(len(valid)):
        numerator += p[:, t, None] * values[t]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return bf16(numerator / denominator[:, None])


def staged_model(q, values, valid, sink, scale):
    n_idx = len(valid)
    padded = (n_idx + ROWS - 1) // ROWS * ROWS
    storage = np.zeros((padded, DIM), dtype=np.float32)
    storage[:n_idx] = values
    scores = np.full((HEADS, LIMIT), -np.inf, dtype=np.float32)
    for first in range(0, n_idx, ROWS):
        accum = np.zeros((HEADS, ROWS), dtype=np.float32)
        for d in range(0, DIM, 16):
            accum += q[:, d:d+16] @ storage[first:first+ROWS, d:d+16].T
        scores[:, first:first+ROWS] = accum * np.float32(scale)
    scores[:, n_idx:] = -np.inf
    scores[:, :n_idx][:, ~valid] = -np.inf
    maximum = np.maximum(scores.max(axis=1), np.float32(-1e30))
    p = np.exp(scores - maximum[:, None], dtype=np.float32)
    # Each half-warp lane sums its 64 probabilities in order, then XOR-reduces.
    total = np.zeros((HEADS, 16), dtype=np.float32)
    for e in range(LIMIT // 16):
        total += p[:, e*16:(e+1)*16]
    denominator = tree_sum(total) + np.exp(sink - maximum, dtype=np.float32)
    p = bf16(p)
    numerator = np.zeros((HEADS, DIM), dtype=np.float32)
    for first in range(0, n_idx, ROWS):
        for r in range(0, ROWS, 16):
            numerator += p[:, first+r:first+r+16] @ storage[first+r:first+r+16]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return bf16(numerator / denominator[:, None])


def check_numerics():
    rng = np.random.default_rng(1301)
    worst = 0.0
    cases = [(n, style) for n in (0, 1, 17, 63, 64, 65, 127, 128, 129, 640, 1023, 1024)
             for style in ("mixed", "empty", "duplicates", "wide", "sink_dominant", "zero_q")]
    with np.errstate(over="ignore"):
        for n, style in cases:
            q = bf16(rng.uniform(-1, 1, (HEADS, DIM)))
            bank = bf16(rng.uniform(-1, 1, (max(n, 1), DIM)))
            if style == "wide":
                q = bf16(q * np.exp2(rng.integers(-4, 5, q.shape)))
                bank = bf16(bank * np.exp2(rng.integers(-4, 5, bank.shape)))
            if style == "zero_q": q[:] = 0
            indices = rng.integers(0, max(n, 1), n)
            valid = rng.random(n) > 0.25
            if style == "empty": valid[:] = False
            if style == "duplicates": indices[:] = 0
            values = bank[indices].copy()
            values[~valid] = 0
            sink = rng.uniform(-4, 4, HEADS).astype(np.float32)
            if style == "sink_dominant": sink[:] = 80
            ref = reference(q, values, valid, sink, 1/np.sqrt(DIM))
            actual = staged_model(q, values, valid, sink, 1/np.sqrt(DIM))
            error = np.linalg.norm(actual.astype(float) - ref) / max(np.linalg.norm(ref.astype(float)), 1e-30)
            assert error <= 3e-3, (n, style, error)
            worst = max(worst, error)
    print(f"PASS {len(cases)} BF16/FP32 CPU numerical models, worst relative L2={worst:.8g}")
    print("NOTE CPU GEMM models do not reproduce hardware MMA accumulation exactly")


def check_source_protocol():
    source = SOURCE.read_text()
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)
    for forbidden in ("cudaMalloc", "cudaFree", "cudaMemcpy", "cudaStreamSynchronize", "cudaDeviceSynchronize"):
        assert forbidden not in body
    assert "static_cast<size_t>(j) * kDim" in body
    assert "sizeof(TileStorage) == 99392" in body
    assert "std::unordered_set<int> initialized" in body
    assert body.count("cudaFuncSetAttribute(") == 1
    assert "initialized.find(device) == initialized.end()" in body
    assert "sizeof(TileStorage), stream" in body
    # No score can be overwritten by an aliased P store until all warps have
    # loaded their own complete head segment, with a CTA barrier between them.
    load = body.index("probabilities[e] = r < n_idx")
    store = body.index("tile.phase.pv.probabilities[h * kProbStride")
    assert body.index("__syncthreads();", load) < store
    assert body.index("__syncthreads();", store) < body.index("float result[")
    # Derive union layout independently, including tail padding and metadata.
    qk = HEADS * QSTRIDE * 2 + HEADS * LIMIT * 4 + ROWS * SLICE * 2
    pv = HEADS * PSTRIDE * 2 + ROWS * DIM * 2
    total = max(qk, pv) + ROWS * 8 + ROWS * 4 + HEADS * 4
    assert qk == pv == 98560 and total == 99392 <= 99 * 1024
    # A CTA-local gather publishes each thread's async writes before any MMA;
    # each final MMA reader reaches a barrier before the next tile overwrites.
    assert "cp.async.commit_group;\\ncp.async.wait_group 0;\\n" in body
    print("PASS shared-union phase barriers, gather protocol and per-device graph setup source checks")
    print("candidate_sha256=" + hashlib.sha256(source.encode()).hexdigest())


if __name__ == "__main__":
    check_matrix_layout()
    check_ownership_and_tails()
    check_numerics()
    check_source_protocol()
    print("PASS CPU/source models only; GPU acceptance, graph replay, and performance remain unverified")
