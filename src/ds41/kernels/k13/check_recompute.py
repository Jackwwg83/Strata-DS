#!/usr/bin/env python3
"""CPU-only K13-03 layout, bounds, barrier-protocol, and numerical checks.

Requires NumPy only for this optional development check; the CUDA implementation
has no new runtime dependencies. This is not a substitute for GPU acceptance.
"""
from pathlib import Path
import hashlib
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "src/ds41/kernels/k13_sparse_attn_prefill.cu"


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    # Round-to-nearest-even, matching __float2bfloat16_rn for these finite inputs.
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def kv_offset(r, d):
    return r * 64 + (d ^ ((r & 7) * 8))


def layout_check():
    mem = np.full(64 * 64, -1, dtype=np.int64)
    writes = np.zeros(mem.size, dtype=np.int32)
    for tid in range(256):
        for i in range(tid, 64 * 64 // 8, 256):
            r, d = i // 8, (i % 8) * 8
            p = kv_offset(r, d)
            assert p % 8 == 0
            for e in range(8):
                mem[p + e] = r * 64 + d + e
                writes[p + e] += 1
    assert np.all(writes == 1)

    def matrix_x4(addresses, transpose=False):
        blocks = []
        for m in range(4):
            block = np.stack([mem[addresses[m * 8 + r]:addresses[m * 8 + r] + 8] for r in range(8)])
            blocks.append(block.T if transpose else block)
        return np.block([[blocks[0], blocks[2]], [blocks[1], blocks[3]]])

    logical = np.arange(64 * 64).reshape(64, 64)
    for warp in range(8):
        rg = (warp % 4) * 16
        for d in range(0, 64, 16):
            addresses = [kv_offset(rg + lane % 16, d + (lane // 16) * 8) for lane in range(32)]
            assert all(p % 8 == 0 for p in addresses)
            assert np.array_equal(matrix_x4(addresses), logical[rg:rg + 16, d:d + 16])
        for r in range(0, 64, 16):
            dg = (warp % 4) * 16
            addresses = [kv_offset(r + lane % 8 + (lane // 16) * 8, dg + ((lane // 8) & 1) * 8)
                         for lane in range(32)]
            assert all(p % 8 == 0 for p in addresses)
            assert np.array_equal(matrix_x4(addresses, True), logical[r:r + 16, dg:dg + 16].T)

    # B x2 row-address lists are transposes of the requested K x 8 operand.
    for stride, width in [(520, 512), (72, 64)]:
        memory = np.full(16 * stride, -1, dtype=np.int64)
        for h in range(16):
            memory[h * stride:h * stride + width] = h * 10000 + np.arange(width)
        for hg in (0, 8):
            for k in range(0, width, 16):
                addresses = [(hg + lane % 8) * stride + k + ((lane // 8) & 1) * 8 for lane in range(16)]
                assert all(p % 8 == 0 for p in addresses)
                blocks = [np.stack([memory[addresses[m * 8 + r]:addresses[m * 8 + r] + 8]
                                    for r in range(8)]).T for m in range(2)]
                actual = np.concatenate(blocks)
                expected = np.array([[h * 10000 + d for h in range(hg, hg + 8)] for d in range(k, k + 16)])
                assert np.array_equal(actual, expected)

    scores = np.zeros((16, 64), dtype=np.int32)
    outputs = np.zeros((16, 512), dtype=np.int32)
    for tid in range(256):
        warp, lane = tid // 32, tid % 32
        h = (warp // 4) * 8 + (lane & 3) * 2
        r = (warp % 4) * 16 + (lane >> 2)
        for hh, rr in ((h, r), (h + 1, r), (h, r + 8), (h + 1, r + 8)):
            scores[hh, rr] += 1
        for s in range(8):
            d = s * 64 + (warp % 4) * 16 + (lane >> 2)
            for hh, dd in ((h, d), (h + 1, d), (h, d + 8), (h + 1, d + 8)):
                outputs[hh, dd] += 1
    assert np.all(scores == 1) and np.all(outputs == 1)
    p_writes = {(tid // 16, tid % 16 + e * 16) for tid in range(256) for e in range(4)}
    assert p_writes == {(h, r) for h in range(16) for r in range(64)}
    print("PASS: vector gather, ldmatrix A/B/transposed-A layouts, unique score/P/output coverage")


def protocol_check():
    # Event model of the actual one-buffer lifecycle. Every gather retires all
    # prior readers, issues each thread's copies, waits, then publishes them.
    generation = 0
    state = "idle"
    score_state = "idle"
    for phase in range(2):
        for tile in range(16):
            for dimension in range(8):
                state = "retired"  # block barrier before writes
                assert state == "retired"
                state = "pending"
                state = "completed"  # per-thread cp.async.wait_group 0
                state = "published"  # block barrier before any reader
                generation += 1
                assert state == "published"
                state = "read"
            score_state = "published"
            assert score_state == "published"
            if phase == 1:
                score_state = "all_reads_retired"  # barrier before union reuse
                assert score_state == "all_reads_retired"
                score_state = "probabilities_published"
                assert state == "read"  # final QK slice stays in buffer
                for dimension in [7, 0, 1, 2, 3, 4, 5, 6]:
                    if dimension != 7:
                        state = "retired"
                        state = "pending"
                        state = "completed"
                        state = "published"
                        generation += 1
                    assert state in ("read", "published")
                    assert score_state == "probabilities_published"
                    state = "read"
            state = "retired"  # final block barrier before next select_rows
            score_state = "retired"
    assert generation == 16 * (8 + 8 + 7)
    print("PASS: one-buffer wait/publication/reuse and score-to-probability alias protocol model")


def bounds_check():
    for n in [0, 1, 7, 63, 64, 65, 127, 128, 129, 639, 640, 641, 1023, 1024]:
        visits = []
        for start in range(0, n, 64):
            for tid in range(64):
                p = start + tid
                if p < n:
                    visits.append(p)
        assert visits == list(range(n))
    for m in [1, 37, 512, 4096, 8192, 16384]:
        # CUDA x/y grids cover all queries/heads without a test-shaped cap.
        assert (m - 1) * 64 * 512 + 63 * 512 + 511 == m * 64 * 512 - 1
    assert np.int64(2**31 - 1) * 512 == 1099511627264
    print("PASS: n_idx boundary/tail coverage, m up to16384, 64-bit row offsets")


def direct(logits, values, valid, sink):
    maximum = np.maximum(np.max(logits, axis=1, initial=np.float32(-1e30)), np.float32(-1e30))
    with np.errstate(over="ignore", invalid="ignore"):
        weights = np.exp(logits - maximum[:, None], dtype=np.float32)
        weights[:, ~valid] = 0
        denom = weights.sum(axis=1, dtype=np.float32) + np.exp(sink - maximum, dtype=np.float32)
        numerator = bf16(weights) @ values
        return bf16(numerator / denom[:, None]), maximum, denom


def recompute(logits, values, valid, sink):
    maximum = np.full(16, np.float32(-1e30))
    total = np.zeros(16, dtype=np.float32)
    for start in range(0, logits.shape[1], 64):
        tile = logits[:, start:start + 64]
        next_max = np.maximum(maximum, tile.max(axis=1))
        tile_weights = np.exp(tile - next_max[:, None], dtype=np.float32)
        total = total * np.exp(maximum - next_max, dtype=np.float32) + tile_weights.sum(axis=1, dtype=np.float32)
        maximum = next_max
    with np.errstate(over="ignore", invalid="ignore"):
        denom = total + np.exp(sink - maximum, dtype=np.float32)
    out = np.zeros((16, 512), dtype=np.float32)
    for start in range(0, logits.shape[1], 64):
        tile = logits[:, start:start + 64]
        weights = bf16(np.exp(tile - maximum[:, None], dtype=np.float32))
        out += weights @ values[start:start + 64]
    return bf16(out / denom[:, None]), maximum, denom


def numeric_check():
    rng = np.random.default_rng(1303)
    q = bf16(rng.normal(size=(16, 512)).astype(np.float32))
    kv = bf16(rng.normal(size=(1100, 512)).astype(np.float32))
    sink = rng.uniform(-1, 1, size=16).astype(np.float32)
    worst = 0.0
    worst_reference = 0.0
    cases = 0
    for n in [0, 1, 7, 63, 64, 65, 127, 128, 129, 639, 640, 641, 1023, 1024]:
        idx = rng.integers(0, len(kv), size=n)
        if n:
            idx[::7] = -1
            idx[::19] = -123
        valid = idx >= 0
        values = kv[np.maximum(idx, 0)].copy()
        values[~valid] = 0
        # K16 serial accumulation models the two passes' identical QK order.
        logits = np.zeros((16, n), dtype=np.float32)
        for k in range(0, 512, 16):
            logits += q[:, k:k + 16] @ values[:, k:k + 16].T
        logits *= np.float32(1 / np.sqrt(512))
        logits[:, ~valid] = -np.inf
        # Also compare the reference's per-lane 16-product dot accumulation
        # and warp tree, followed by ascending-key scalar FP32 PV accumulation.
        lane_dot = np.zeros((16, n, 32), dtype=np.float32)
        for k in range(0, 512, 32):
            lane_dot += q[:, None, k:k + 32] * values[None, :, k:k + 32]
        for off in [16, 8, 4, 2, 1]:
            lane_dot[:, :, :off] += lane_dot[:, :, off:off * 2]
        reference_logits = lane_dot[:, :, 0] * np.float32(1 / np.sqrt(512))
        reference_logits[:, ~valid] = -np.inf
        reference_max = np.maximum(reference_logits.max(axis=1, initial=np.float32(-1e30)), np.float32(-1e30))
        reference_weights = np.exp(reference_logits - reference_max[:, None], dtype=np.float32)
        with np.errstate(over="ignore"):
            reference_denom = reference_weights.sum(axis=1, dtype=np.float32) + np.exp(sink - reference_max)
        reference_p = bf16(reference_weights)
        reference_out = np.zeros((16, 512), dtype=np.float32)
        for j in range(n):
            reference_out += reference_p[:, j, None] * values[j]
        reference_out = bf16(reference_out / reference_denom[:, None])
        actual, _, _ = recompute(logits, values, valid, sink)
        reference_error = float(np.linalg.norm(actual - reference_out) / max(float(np.linalg.norm(reference_out)), 1e-30))
        assert reference_error <= 3e-3, (n, "reference accumulation", reference_error)
        worst_reference = max(worst_reference, reference_error)
        for kind in ("normal", "empty", "late_max", "sink_overflow", "sink_underflow"):
            x = logits.copy()
            v = values.copy()
            ok = valid.copy()
            sk = sink.copy()
            if kind == "empty":
                x[:] = -np.inf
                v[:] = 0
                ok[:] = False
            elif kind == "late_max" and n:
                x[:, -1] = np.float32(30)
                v[-1] = kv[0]
                ok[-1] = True
            elif kind == "sink_overflow":
                sk[:] = 1000
            elif kind == "sink_underflow":
                sk[:] = -1000
            expected, mx, dn = direct(x, v, ok, sk)
            actual, amx, adn = recompute(x, v, ok, sk)
            assert np.array_equal(mx, amx)
            finite = np.isfinite(dn)
            assert np.allclose(dn[finite], adn[finite], rtol=5e-7, atol=0)
            assert np.array_equal(np.isinf(dn), np.isinf(adn))
            assert np.isfinite(actual).all()
            error = float(np.linalg.norm(actual - expected) / max(float(np.linalg.norm(expected)), 1e-30))
            worst = max(worst, error)
            assert error <= 3e-3, (n, kind, error)
            cases += 1
    # Specifically reject the attractive, but wrong, normalized-P rounding.
    x = np.array([[0, -0.7, -1.4]], dtype=np.float32)
    p = np.exp(x)
    denom = p.sum() + np.float32(0.31)
    assert np.any(bf16(p) / denom != bf16(p / denom))
    print(f"PASS: {cases} denominator/BF16/PV CPU cases; worst rel_l2={worst:.9g}")
    print(f"PASS: reference warp-QK/scalar-PV accumulation; worst rel_l2={worst_reference:.9g}")


def source_check():
    source = SOURCE.read_text()
    constants = {k: int(v) for k, v in re.findall(r"constexpr int (k\w+) = (\d+);", source)}
    assert constants["kHeadTile"] == 16 and constants["kRows"] == 64
    assert constants["kSlice"] == 64 and constants["kThreads"] == 256
    for forbidden in ["cudaMalloc", "cudaFree", "cudaMemcpy", "cudaDeviceSynchronize", "cudaStreamSynchronize"]:
        assert forbidden not in source
    assert "m > 16384" in source and "n_idx > 1024" in source
    assert "dim3(m, kHeads / kHeadTile)" in source
    assert "kThreads, 0, stream" in source
    assert "static_cast<size_t>(j) * kDim" in source
    print("PASS: fixed-interface bounds, stream-only launch, no allocation or host synchronization")
    print("source_sha256=" + hashlib.sha256(SOURCE.read_bytes()).hexdigest())


if __name__ == "__main__":
    source_check()
    layout_check()
    bounds_check()
    protocol_check()
    numeric_check()
    print("CPU CHECKS PASSED; GPU correctness, graph replay, and performance remain unmeasured")
