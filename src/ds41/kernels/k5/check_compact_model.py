"""CPU model for K5-04 compaction/selection. Does not execute CUDA or time it."""
import math
import random
import struct


def order_key(score, position):
    bits = struct.unpack('<I', struct.pack('<f', 0.0 if score == 0 else score))[0]
    bits ^= 0xffffffff if bits & 0x80000000 else 0x80000000
    return (bits << 32) | (0xffffffff - position)


def select(scores, positions, k, bound):
    if not k:
        return [], 0
    capacity = 1
    while capacity < min(bound, k * 2):
        capacity *= 2
    values = [order_key(score, pos) for score, pos in zip(scores, positions)]
    assert len(values) >= k
    threshold = 0
    iterations = 0
    if len(values) > capacity:
        low, high = min(values), max(values)
        while True:
            assert low <= high
            pivot = low + ((high - low) >> 1)
            above = [x for x in values if x >= pivot]
            below = [x for x in values if x < pivot]
            iterations += 1
            assert iterations <= 65
            if k <= len(above) <= capacity:
                threshold = pivot
                break
            if len(above) < k:
                high = max(below)
            else:
                low = min(above) + 1
    retained = [x for x in values if x >= threshold]
    assert k <= len(retained) <= capacity
    retained.sort(reverse=True)
    return sorted(0xffffffff - (x & 0xffffffff) for x in retained[:k]), iterations


def compact(scores, mask, k, rng):
    # Simulate arbitrary completion order of CTAs and stable ballot compaction
    # within each CTA. Original-position tie keys make CTA order irrelevant.
    blocks = list(range(0, len(scores), 256))
    rng.shuffle(blocks)
    positions = [i for b in blocks for i in range(b, min(b + 256, len(scores)))
                 if mask[i] or i < k]
    compact_scores = [scores[i] if mask[i] else -math.inf for i in positions]
    assert len(positions) >= k
    assert len(positions) == len(set(positions))
    assert set(positions) == {i for i in range(len(scores)) if mask[i] or i < k}
    return compact_scores, positions


def bitonic(values, descending):
    width = 2
    while width <= len(values):
        stride = width >> 1
        while stride:
            touched = set()
            for i in range(len(values)):
                j = i ^ stride
                if j > i:
                    assert i not in touched and j not in touched
                    touched.update((i, j))
                    down = ((i & width) == 0) == descending
                    if values[i] < values[j] if down else values[i] > values[j]:
                        values[i], values[j] = values[j], values[i]
            stride >>= 1
        width <<= 1
    return values


def main():
    rng = random.Random(504)
    runs = max_passes = 0
    sizes = (1, 2, 31, 32, 33, 255, 256, 257, 300, 511, 512, 513,
             1023, 1024, 1025, 4095, 4096, 4097, 16384, 131072)
    for n in sizes:
        scoresets = [[-math.inf] * n, [0.0] * n,
                     [rng.choice((-math.inf, -3., -0., 0., 1., 3., math.inf)) for _ in range(n)],
                     [float(i // 11) for i in range(n)]]
        masks = [[0] * n, [1] * n, [int(rng.random() < .75) for _ in range(n)],
                 [int(i % 97 == 0) for i in range(n)]]
        for k in sorted({min(n, k) for k in (0, 1, 3, 64, 511, 512, 513, 2049, n)}):
            for scores in scoresets:
                for mask in masks:
                    cs, ps = compact(scores, mask, k, rng)
                    got, passes = select(cs, ps, k, n)
                    full = [s if m else -math.inf for s, m in zip(scores, mask)]
                    want = sorted(sorted(range(n), key=lambda i: (-full[i], i))[:k])
                    assert got == want, (n, k, got[:10], want[:10])
                    max_passes = max(max_passes, passes)
                    runs += 1
    for _ in range(1000):
        n = rng.randrange(1, 4098)
        k = rng.randrange(n + 1)
        scores = [rng.choice((-math.inf, -0., 0., 1., rng.gauss(0, 10))) for _ in range(n)]
        mask = [int(rng.random() < rng.random()) for _ in range(n)]
        cs, ps = compact(scores, mask, k, rng)
        got, passes = select(cs, ps, k, n)
        full = [s if m else -math.inf for s, m in zip(scores, mask)]
        want = sorted(sorted(range(n), key=lambda i: (-full[i], i))[:k])
        assert got == want
        max_passes = max(max_passes, passes)
        runs += 1
    print(f'PASS compact exact selection: {runs} cases; max threshold passes={max_passes}')
    for capacity in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        values = [rng.getrandbits(64) for _ in range(capacity)]
        for down in (False, True):
            assert bitonic(values[:], down) == sorted(values, reverse=down)
        for k in sorted({1, max(1, capacity // 2), capacity}):
            ranked = sorted(values, reverse=True)
            output_capacity = 1
            while output_capacity < k:
                output_capacity *= 2
            kept = [(0xffffffff - (v & 0xffffffff)) if i < k else 0xffffffffffffffff
                    for i, v in enumerate(ranked)]
            got = bitonic(kept[:output_capacity], False)[:k]
            assert got == sorted(0xffffffff - (v & 0xffffffff) for v in ranked[:k])
    print('PASS bitonic selection and reduced output sort: capacities 1..8192')
    count = 0
    for n in (1, 7, 8, 9, 300, 16384):
        for block in (1, 3, 8, 13):
            for k in (1, 2, 64, 10000):
                scores = [rng.choice((-math.inf, -0., 0., 1., 3., math.inf)) for _ in range(n)]
                maxima = [max(scores[i:i + block]) for i in range(0, n, block)]
                maxima[-1] = math.inf
                keep = min(k, len(maxima))
                selected, _ = select(maxima, list(range(len(maxima))), keep, len(maxima))
                expected = sorted(range(len(maxima)), key=lambda b: (-maxima[b], b))[:keep]
                def emit(indices):
                    cand = [0] * n
                    for b in indices:
                        if maxima[b] != -math.inf:
                            cand[b * block:min(n, (b + 1) * block)] = [1] * min(block, n - b * block)
                    return cand
                assert emit(selected) == emit(expected)
                count += 1
    print(f'PASS exact candidate blocks: {count} cases with tails, infinities and signed zero')
    print('CUDA correctness, stream execution and timing require the GPU queue')


if __name__ == '__main__':
    main()
