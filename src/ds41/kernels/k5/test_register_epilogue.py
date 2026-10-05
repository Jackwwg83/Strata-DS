#!/usr/bin/env python3
"""K5-17 CPU ownership/order proof; no CUDA math or races are simulated."""
import math
from pathlib import Path
import random
import struct


def f32(v):
    return struct.unpack('<f', struct.pack('<f', v))[0]


def bits(v):
    return struct.unpack('<I', struct.pack('<f', v))[0]


def value(v):
    return struct.unpack('<f', struct.pack('<I', v))[0]


def bf(v):
    u = bits(f32(v))
    if u & 0x7f800000 == 0x7f800000:
        return value(u & 0xffff0000)
    return value((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000)


def coordinate(lane, r):
    return (16 * (r >> 3) + 8 * ((r >> 1) & 1) + (lane >> 2),
            8 * ((r >> 2) & 1) + 2 * (lane & 3) + (r & 1))


def transpose(v):
    # Literal model of transpose_bit. Shuffles read the source lane's chosen
    # value, not the destination lane's predicate applied at the source.
    for lane_bit, register_bit in ((16, 8), (8, 4), (4, 1)):
        for r in range(16):
            if r & register_bit:
                continue
            chosen = [row[r] if lane & lane_bit else row[r | register_bit]
                      for lane, row in enumerate(v)]
            for lane in range(32):
                cross = chosen[lane ^ lane_bit]
                if lane & lane_bit:
                    v[lane][r] = cross
                else:
                    v[lane][r | register_bit] = cross
    return v


def reg(h):
    return (h & 1) | ((h & 8) >> 2) | ((h & 6) << 1)


def key(lane):
    return ((lane >> 3) & 1) * 8 + (lane & 3) * 2 + ((lane >> 2) & 1)


def check_operands():
    # An ordinary non-transposed ldmatrix assigns a pair of columns at 2*t
    # to lane 4*g+t in each 8x8 matrix. Matrix m's rows use supplying lanes
    # 8*m+g. Check the staged Q/K address against the PTX bf16 fragment maps.
    for d in range(0, 128, 16):
        for head_half in range(2):
            seen = set()
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for r in range(4):
                    supplier = r * 8 + g
                    address = 16 * head_half * 128 + (supplier & 15) * 128 + (supplier >> 4) * 8 + d
                    assert (address * 2) % 16 == 0
                    for pair in range(2):
                        physical = address + 2 * t + pair
                        scalar = 2 * r + pair
                        expected_h = 16 * head_half + g + (8 if scalar % 4 >= 2 else 0)
                        expected_d = d + 2 * t + pair + (8 if scalar >= 4 else 0)
                        assert physical == expected_h * 128 + expected_d
                        assert 0 <= physical < 32 * 128
                        seen.add((expected_h, expected_d))
            assert len(seen) == 16 * 16
        for warp in range(4):
            for key_half in range(2):
                seen = set()
                for lane in range(32):
                    g, t = lane >> 2, lane & 3
                    for r in range(2):
                        supplier = r * 8 + g
                        address = (warp * 16 + key_half * 8 + (supplier & 7)) * 128 + ((supplier >> 3) & 1) * 8 + d
                        assert (address * 2) % 16 == 0
                        for pair in range(2):
                            physical = address + 2 * t + pair
                            expected_key = warp * 16 + key_half * 8 + g
                            expected_d = d + 2 * t + pair + 8 * r
                            assert physical == expected_key * 128 + expected_d
                            assert 0 <= physical < 64 * 128
                            seen.add((expected_key, expected_d))
                assert len(seen) == 8 * 16
    print('PASS PTX A/B fragment mapping and aligned ldmatrix addresses for all 8 K steps')


def check_transpose():
    source = [[coordinate(lane, r) for r in range(16)] for lane in range(32)]
    assert len({x for row in source for x in row}) == 32 * 16
    result = transpose(source)
    written = set()
    for lane in range(32):
        for h in range(16):
            assert result[lane][reg(h)] == ((lane >> 4) * 16 + h, key(lane))
        if lane >= 16:
            assert key(lane) not in written
            written.add(key(lane))
            order = [result[lane - 16][reg(h)][0] for h in range(16)]
            order += [result[lane][reg(h)][0] for h in range(16)]
            assert order == list(range(32))
    assert written == set(range(16))
    print('PASS all 512 accumulator coordinates, bijective register transpose, exact h0..31 order')


def epilogue(dots, weights):
    contributions = [[bf(f32(max(bf(dots[h][j]), 0.0) * weights[h]))
                      for j in range(16)] for h in range(32)]
    v = transpose([[contributions[h][j] for h, j in
                   (coordinate(lane, r) for r in range(16))] for lane in range(32)])
    carry = [0.0] * 16
    for lane in range(16):
        for h in range(16):
            carry[lane] = f32(carry[lane] + v[lane][reg(h)])
    got = [None] * 16
    for lane in range(16, 32):
        total = carry[lane & 15]
        for h in range(16):
            total = f32(total + v[lane][reg(h)])
        got[key(lane)] = bits(bf(total))
    want = []
    for j in range(16):
        total = 0.0
        for h in range(32):
            total = f32(total + contributions[h][j])
        want.append(bits(bf(total)))
    assert got == want


def check_numerics():
    rng = random.Random(170032)
    for trial in range(256):
        weights = [bf(rng.uniform(-8, 8)) for _ in range(32)]
        dots = [[f32(rng.uniform(-128, 128)) for _ in range(16)] for _ in range(32)]
        epilogue(dots, weights)
    # Distinguishes the ordered carry from adding independently reduced halves:
    # h0=2^25, h15=1, h16=-2^25, h17=1 => ordered=1, partial-halves=0.
    weights = [bf(0)] * 32
    weights[0], weights[15], weights[16], weights[17] = bf(1), bf(1), bf(-1), bf(1)
    dots = [[0.0] * 16 for _ in range(32)]
    for j in range(16):
        dots[0][j] = dots[16][j] = float(2**25)
        dots[15][j] = dots[17][j] = 1.0
    epilogue(dots, weights)
    ordered = f32(f32(f32(f32(0 + 2**25) + 1) - 2**25) + 1)
    partials = f32(f32(2**25 + 1) + f32(-2**25 + 1))
    assert ordered != partials
    print('PASS 4096 randomized exact BF16 weighted/order outputs and cancellation witness')


def check_tails():
    for n in (513, 514, 527, 528, 575, 576, 577, 4095, 4096, 4097, 16384, 131072, 131201):
        covered = bytearray(n)
        for base in range(0, n, 64):
            for warp in range(4):
                for lane in range(16, 32):
                    j = base + warp * 16 + key(lane)
                    if j < n:
                        covered[j] += 1
        assert all(x == 1 for x in covered)
    print('PASS unique writers for all keys and irregular 64-key tensor tails')


def main():
    check_operands()
    check_transpose()
    check_numerics()
    check_tails()


if __name__ == '__main__':
    main()
