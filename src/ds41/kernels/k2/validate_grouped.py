#!/usr/bin/env python3
"""CPU-only structural model for K2-06. This does not replace the GPU test."""
from collections import Counter
import math
import struct

SHAPES = [(16, 64, 1), (32, 64, 1), (64, 128, 2)]


def schedule(pid, tiles_m, tiles_n):
    first_m = (pid // (4 * tiles_n)) * 4
    group_m = min(tiles_m - first_m, 4)
    within = pid % (4 * tiles_n)
    return first_m + within % group_m, within // group_m


def output_coordinates(tm, tn, groups):
    for tid in range(tn * 2):
        warp, lane = divmod(tid, 32)
        warp_m = (warp // (tn // tm)) * 16
        warp_n = (warp % (tn // tm)) * tm
        for group in range(groups):
            for j in range(tm // 8):
                for i in range(4):
                    yield (group * tm + warp_m + lane // 4 + (i // 2) * 8,
                           warp_n + j * 8 + (lane % 4) * 2 + i % 2)


def check_layout():
    for tm, tn, groups in SHAPES:
        a, b = Counter(), Counter()
        for tid in range(tn * 2):
            for group in range(groups):
                if tid < tm * 4:
                    for k in range(8):
                        a[group * tm + tid // 4, (tid % 4) * 8 + k] += 1
            for k in range(16):
                b[tid // 2, (tid % 2) * 16 + k] += 1
        assert a == Counter({(m, k): 1 for m in range(groups * tm) for k in range(32)})
        assert b == Counter({(n, k): 1 for n in range(tn) for k in range(32)})
        out = Counter(output_coordinates(tm, tn, groups))
        assert out == Counter({(m, n): 1 for m in range(tm * groups) for n in range(tn)})
        assert (groups * tm + tn) * 40 * 2 <= 99 * 1024
        # Shared row starts and all eight-element matrix addresses are aligned.
        for tid in range(tn * 2):
            warp, lane = divmod(tid, 32)
            wm, wn = (warp // (tn // tm)) * 16, (warp % (tn // tm)) * tm
            for k in (0, 16):
                for group in range(groups):
                    offset = (group * tm + wm + lane % 16) * 40 + k + (lane // 16) * 8
                    assert offset % 8 == 0 and offset + 7 < groups * tm * 40
                for j in range(tm // 8):
                    offset = (wn + j * 8 + lane % 8) * 40 + k + ((lane // 8) % 2) * 8
                    assert offset % 8 == 0 and offset + 7 < tn * 40
    print('PASS shared/vector layout, MMA output ownership, alignment, shared-memory bounds')


def check_schedule():
    count = 0
    for tiles_m in list(range(1, 35)) + [63, 64, 65, 127, 128, 129, 1024]:
        for tiles_n in list(range(1, 35)) + [63, 64, 65, 127, 128, 129, 256]:
            got = {schedule(pid, tiles_m, tiles_n) for pid in range(tiles_m * tiles_n)}
            assert got == {(m, n) for m in range(tiles_m) for n in range(tiles_n)}
            count += 1
    print('PASS grouped schedule bijection:', count, 'tile grids, including partial supergroups')


def check_tails_and_products():
    # Independent integer-valued operands make these sums exactly representable
    # as FP32. Accumulate in the modeled 16-wide MMA reduction order.
    for tm, tn, groups in SHAPES:
        for M, N, K in [(1, 1, 0), (13, 7, 32), (77, 65, 64), (129, 129, 96)]:
            A = [[(m * 5 + k * 3) % 11 - 5 for k in range(K)] for m in range(M)]
            W = [[(n * 7 + k * 2) % 13 - 6 for k in range(K)] for n in range(N)]
            got = {}
            mt, nt = (M + groups * tm - 1) // (groups * tm), (N + tn - 1) // tn
            coordinates = tuple(output_coordinates(tm, tn, groups))
            for pid in range(mt * nt):
                tile_m, tile_n = schedule(pid, mt, nt)
                for local_m, local_n in coordinates:
                    m, n = tile_m * groups * tm + local_m, tile_n * tn + local_n
                    if m >= M or n >= N:
                        continue
                    assert (m, n) not in got
                    acc = 0
                    for kb in range(0, K, 32):
                        for half in (0, 16):
                            for k in range(kb + half, kb + half + 16):
                                acc += A[m][k] * W[n][k]
                    got[m, n] = acc
            assert len(got) == M * N
            for (m, n), value in got.items():
                assert value == sum(a * b for a, b in zip(A[m], W[n]))
            # Each activation occupies two bytes, within the fixed four-byte budget.
            assert 2 * M * K <= 4 * M * K
    print('PASS tails, K=0, grouped products and workspace bounds for all three instantiations')


def fp8(b):
    e, mant = (b >> 3) & 15, b & 7
    if e == 15 and mant == 7:
        return math.nan
    value = math.ldexp(mant / 8.0, -6) if e == 0 else math.ldexp(1 + mant / 8.0, e - 7)
    return -value if b & 128 else value


def bf16(v):
    word = struct.unpack('<I', struct.pack('<f', v))[0]
    word = (word + 0x7fff + ((word >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', word))[0]


def check_operand_exactness():
    count = 0
    for byte in range(256):
        value = fp8(byte)
        if math.isnan(value):
            continue
        for exponent in range(-110, 111):
            scaled = math.ldexp(value, exponent)
            assert bf16(scaled) == scaled
            count += 1
    print('PASS', count, 'finite FP8/power-of-two operands are exactly BF16 in tested exponent range')


if __name__ == '__main__':
    check_layout()
    check_schedule()
    check_tails_and_products()
    check_operand_exactness()
    print('CPU model only: device synchronization, instruction semantics, parity and timings require GPU CI')
