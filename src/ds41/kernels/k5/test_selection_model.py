#!/usr/bin/env python3
"""CPU model of K5-07 shared-only radix cutoff and stable positional compaction.

This checks selection mathematics and grid-stride coverage only, not CUDA
arithmetic, device synchronization, graph replay, or GPU performance.
"""
import math
import random
import struct


def key(value, bf16):
    bits = 0 if value == 0 else struct.unpack('<I', struct.pack('<f', value))[0]
    flipped = bits ^ (0xffffffff if bits & 0x80000000 else 0x80000000)
    return flipped >> 16 if bf16 else flipped


def select(values, k, bf16):
    n = len(values)
    k = min(max(k, 0), n)
    if not k:
        return []
    keys = [key(x, bf16) for x in values]
    prefix, mask, remaining = 0, 0, k
    for shift in range(8 if bf16 else 24, -1, -8):
        bins = [0] * 256
        for v in keys:
            if v & mask == prefix:
                bins[v >> shift & 255] += 1
        for b in range(255, -1, -1):
            if bins[b] >= remaining:
                prefix |= b << shift
                break
            remaining -= bins[b]
        mask |= 255 << shift
    out, greater_before, equal_before = [None] * k, 0, 0
    for base in range(0, n, 1024):
        counts = [(sum(v > prefix for v in keys[j:min(j+32, n)]),
                   sum(v == prefix for v in keys[j:min(j+32, n)]))
                  for j in range(base, base + 1024, 32)]
        greater, equal = 0, 0
        for chunk, (own_greater, own_equal) in enumerate(counts):
            lane_greater, lane_equal = 0, 0
            for j in range(base + chunk * 32, min(n, base + (chunk + 1) * 32)):
                preceding_greater = greater_before + greater + lane_greater
                preceding_equal = equal_before + equal + lane_equal
                if keys[j] > prefix or (keys[j] == prefix and preceding_equal < remaining):
                    out[preceding_greater + min(preceding_equal, remaining)] = j
                lane_greater += keys[j] > prefix
                lane_equal += keys[j] == prefix
            greater += own_greater
            equal += own_equal
        greater_before += greater
        equal_before += equal
    return out


def reference(values, k):
    return sorted(sorted(range(len(values)), key=lambda j: (-values[j], j))[:max(0, k)])


def candidate_scores(values, block):
    result = [max(values[i:i+block]) for i in range(0, len(values), block)]
    result[-1] = math.inf
    return result


def main():
    rng = random.Random(5005)
    cases = 0
    for n in (1, 31, 127, 128, 129, 255, 256, 300, 511, 512, 513,
              1023, 1024, 1025, 16384, 131072, 131201, 262273):
        # The fused score CTA loop must cover all keys exactly once, including
        # a tail tile after the fixed partition count has saturated.
        tiles = min(256, (n + 31) // 32)
        visits = [0] * n
        for tile in range(tiles):
            for base in range(tile * 32, n, tiles * 32):
                for j in range(base, min(n, base + 32)):
                    visits[j] += 1
        assert all(v == 1 for v in visits)
        populations = [
            [0.0] * n,
            [-math.inf] * n,
            [rng.choice([-math.inf, -8.0, -0.0, 0.0, 0.25, 1.0, 2.0, math.inf]) for _ in range(n)],
        ]
        for values in populations:
            for k in sorted(set((0, 1, min(17, n), min(512, n), n))):
                assert select(values, k, True) == reference(values, k), (n, k)
                cases += 1
        # Full FP32 candidate maxima must not be truncated to BF16. Also cover
        # non-default block sizes and partial final blocks.
        values = [rng.uniform(-1, 1) for _ in range(n)]
        values = [struct.unpack('<f', struct.pack('<f', x))[0] for x in values]
        for block in (1, 3, 8, 17):
            maxima = candidate_scores(values, block)
            for k in (1, min(64, len(maxima)), len(maxima)):
                assert select(maxima, k, False) == reference(maxima, k), (n, block, k)
                cases += 1
    print(f'PASS {cases} radix/tie/candidate/partition cases (CPU model only)')


if __name__ == '__main__':
    main()
