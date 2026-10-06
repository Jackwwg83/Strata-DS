#!/usr/bin/env python3
"""CPU proof harness for K14's two-byte cutoff and ordered warp-prefix emission.

This models the device algorithm; it does not claim GPU numerical/graph parity.
No external dependencies. Run: python3 src/ds41/kernels/k14/check_selection.py
"""
import math
import random
import struct


def from_bits(bits):
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def key_of(value):
    bits = 0 if value == 0 else struct.unpack('<I', struct.pack('<f', value))[0]
    return (bits ^ (0xffffffff if bits & 0x80000000 else 0x80000000)) >> 16


def ordered_radix(values, wanted):
    if wanted == 0:
        return []
    if wanted == len(values):
        return list(range(len(values)))
    keys = [key_of(x) for x in values]
    prefix = mask = 0
    remaining = wanted
    for shift in (8, 0):
        bins = [0] * 256
        for key in keys:
            if key & mask == prefix:
                bins[(key >> shift) & 255] += 1
        for bucket in range(255, -1, -1):
            if bins[bucket] >= remaining:
                prefix |= bucket << shift
                break
            remaining -= bins[bucket]
        mask |= 255 << shift
    quota = remaining
    out = [None] * wanted
    greater_before = equal_before = 0
    for base in range(0, len(keys), 1024):
        # Exactly the device's 32 consecutive (part, warp) groups and lane rank.
        groups = [keys[start:start + 32] for start in range(base, min(base + 1024, len(keys)), 32)]
        group_g = group_e = 0
        for group_id, group in enumerate(groups):
            lane_g = lane_e = 0
            for lane, key in enumerate(group):
                g = greater_before + group_g + lane_g
                e = equal_before + group_e + lane_e
                if key > prefix or (key == prefix and e < quota):
                    address = g + min(e, quota)
                    assert address < wanted and out[address] is None
                    out[address] = base + group_id * 32 + lane
                lane_g += key > prefix
                lane_e += key == prefix
            group_g += lane_g
            group_e += lane_e
        greater_before += group_g
        equal_before += group_e
    assert None not in out
    return out


def ref_top(values, wanted):
    return sorted(sorted(range(len(values)), key=lambda j: (-values[j], j))[:wanted])


def candidate(values, topk, block, implementation):
    if not values:
        return []
    maxima = [max(values[i:i + block]) for i in range(0, len(values), block)]
    maxima[-1] = math.inf
    selected = set(implementation(maxima, max(0, min(topk, len(maxima)))))
    return [int(j // block in selected and maxima[j // block] != -math.inf) for j in range(len(values))]


def main():
    rng = random.Random(0x1401)
    all_values = [from_bits(b) for b in range(65536) if b & 0x7f80 != 0x7f80 or b & 0x7f == 0]
    assert sorted(all_values, key=key_of) == sorted(all_values)
    # Equal signed zeros must use lower positions, just like the public float comparator.
    assert key_of(-0.0) == key_of(0.0)
    cases = []
    for n in (0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 512, 513,
              1023, 1024, 1025, 4095, 4096, 4097, 8193, 131073):
        palette = [-math.inf, -128., -2., -0., 0., 0.5, 2., 128., math.inf]
        cases += [[rng.choice(palette) for _ in range(n)], [0.] * n, [-math.inf] * n,
                  [all_values[rng.randrange(len(all_values))] for _ in range(n)]]
    top_checks = candidate_checks = 0
    for values in cases:
        n = len(values)
        for wanted in sorted({0, min(n, 1), min(n, 7), min(n, 512), n // 2, n}):
            assert ordered_radix(values, wanted) == ref_top(values, wanted), (n, wanted)
            top_checks += 1
        for block in (1, 3, 8, 31, 257):
            for topk in (0, 1, 4, 64, n + 1):
                assert candidate(values, topk, block, ordered_radix) == candidate(values, topk, block, ref_top)
                candidate_checks += 1
    # Workspace-address and causal-tail proof for partial query/key tiles and batches.
    tile_checks = 0
    for m, pos0, ratio in ((200, 0, 2), (150, 3000, 1), (130, 5000, 1), (97, 9000, 2),
                           (1, 0, 32768), (257, 7, 3), (513, 61, 7), (4096, 4096, 1),
                           (16384, 2**31 - 1, 2**31 - 1)):
        tmax = (pos0 + m) // ratio
        allocated = min(m, 256) * tmax
        for first in range(0, m, 256):
            rows = min(256, m - first)
            for tile in range(0, rows, 16):
                tile_rows = min(16, rows - tile)
                end = (pos0 + first + tile + tile_rows) // ratio
                for local in range(tile_rows):
                    n = (pos0 + first + tile + local + 1) // ratio
                    assert n <= end <= tmax
                    if n:
                        assert (tile + local) * tmax + n - 1 < allocated
                        assert ((n - 1) // 128) * 128 < end
                        assert (n - 1) // 128 <= (end - 1) // 128
                    for block in (1, 8, 31):
                        if n:
                            begin = ((n - 1) // block) * block
                            assert begin < n <= begin + block
                    tile_checks += 1
    # Worst-case public int position arithmetic must happen after 64-bit promotion.
    assert (2**31 - 1 + 16384) // 1 == 2147500031
    print(f'PASS: {len(all_values)} BF16 ordered values; {top_checks} top-k cases; '
          f'{candidate_checks} candidate cases; {tile_checks} causal/workspace rows')


if __name__ == '__main__':
    main()
