#!/usr/bin/env python3
"""CPU model checks for K5-03's histogram and stable extraction (no GPU claims).

Run: python3 src/ds41/kernels/k5/verify_histogram.py
Only the Python standard library is used. The fixed acceptance test is untouched.
"""
import math
import random
import struct
import unittest

THREADS = 256
TILE = 1024


def from_bits(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def float_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def bf16(value):
    bits = float_bits(value)
    if bits & 0x7f800000 != 0x7f800000:
        bits += 0x7fff + ((bits >> 16) & 1)
    return from_bits(bits & 0xffff0000)


def key(value, width):
    bits = float_bits(value) >> (32 - width)
    sign = 1 << (width - 1)
    if bits & (sign - 1) == 0:
        bits = 0
    return bits ^ ((1 << width) - 1) if bits & sign else bits ^ sign


def scan_pair(values):
    """Emulate the two warp scans, including simultaneous shuffle reads."""
    assert len(values) == THREADS
    lane_values = [list(v) for v in values]
    for delta in (1, 2, 4, 8, 16):
        old = [v[:] for v in lane_values]
        for i in range(THREADS):
            if i % 32 >= delta:
                lane_values[i] = [old[i][c] + old[i - delta][c] for c in (0, 1)]
    warp_totals = [lane_values[w * 32 + 31][:] for w in range(8)]
    for delta in (1, 2, 4):
        old = [v[:] for v in warp_totals]
        for i in range(8):
            if i >= delta:
                warp_totals[i] = [old[i][c] + old[i - delta][c] for c in (0, 1)]
    prefixes = []
    for i in range(THREADS):
        before = warp_totals[i // 32 - 1] if i >= 32 else (0, 0)
        prefixes.append(tuple(before[c] + lane_values[i][c] - values[i][c] for c in (0, 1)))
    return prefixes, tuple(warp_totals[-1])


def histogram_cut(keys, k, width):
    prefix, needed = 0, k
    for shift in range(width - 8, -1, -8):
        mask = ((1 << width) - 1) ^ ((1 << (shift + 8)) - 1)
        bins = [0] * 256
        for value in keys:
            if value & mask == prefix:
                bins[(value >> shift) & 255] += 1
        above = 0
        for byte in range(255, -1, -1):
            if above < needed <= above + bins[byte]:
                prefix |= byte << shift
                needed -= above
                break
            above += bins[byte]
        else:
            raise AssertionError('a nonempty cutoff bin must exist')
    return prefix, needed


def stable_extract(keys, threshold, equal_needed, k, offset):
    tiles = []
    for base in range(0, len(keys), TILE):
        values = keys[base:base + TILE]
        tiles.append((sum(v > threshold for v in values), sum(v == threshold for v in values)))
    prefixes = []
    carry = (0, 0)
    for base in range(0, len(tiles), THREADS):
        chunk = tiles[base:base + THREADS]
        before, total = scan_pair(chunk + [(0, 0)] * (THREADS - len(chunk)))
        prefixes.extend((carry[0] + g, carry[1] + e) for g, e in before[:len(chunk)])
        carry = (carry[0] + total[0], carry[1] + total[1])
    out = [None] * k
    for tile, initial in enumerate(prefixes):
        carry = initial
        for r in range(TILE // THREADS):
            base = tile * TILE + r * THREADS
            values = keys[base:base + THREADS]
            local = [(int(v > threshold), int(v == threshold)) for v in values]
            local += [(0, 0)] * (THREADS - len(local))
            before, total = scan_pair(local)
            for lane, ((g, e), (pg, pe)) in enumerate(zip(local, before)):
                preceding_equal = carry[1] + pe
                if g or (e and preceding_equal < equal_needed):
                    pos = carry[0] + pg + min(preceding_equal, equal_needed)
                    assert 0 <= pos < k and out[pos] is None
                    out[pos] = base + lane + offset
            carry = (carry[0] + total[0], carry[1] + total[1])
    assert None not in out
    return out


def select(scores, k, width=16, offset=0):
    k = max(0, min(k, len(scores)))
    if k == 0 or k == len(scores):
        return list(range(offset, offset + k))
    keys = [key(v, width) for v in scores]
    threshold, needed = histogram_cut(keys, k, width)
    return stable_extract(keys, threshold, needed, k, offset)


def reference(scores, k, offset=0):
    chosen = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:max(0, k)]
    return [i + offset for i in sorted(chosen)]


def candidates(scores, k, block):
    maxima = [max(scores[i:i + block]) for i in range(0, len(scores), block)]
    if not maxima:
        return []
    maxima[-1] = math.inf
    chosen = set(select(maxima, k, 32))
    return [int(i // block in chosen and maxima[i // block] != -math.inf) for i in range(len(scores))]


def candidate_reference(scores, k, block):
    maxima = [max(scores[i:i + block]) for i in range(0, len(scores), block)]
    if not maxima:
        return []
    maxima[-1] = math.inf
    selected = sorted(range(len(maxima)), key=lambda i: (-maxima[i], i))[:max(0, k)]
    result = [0] * len(scores)
    for b in selected:
        if maxima[b] != -math.inf:
            for i in range(b * block, min((b + 1) * block, len(scores))):
                result[i] = 1
    return result


class HistogramModelTest(unittest.TestCase):
    def test_all_finite_bf16_orderings(self):
        values = [from_bits(bits << 16) for bits in range(65536)]
        values = sorted(v for v in values if not math.isnan(v))
        ordered = [key(v, 16) for v in values]
        self.assertEqual(ordered, sorted(ordered))
        self.assertEqual(key(-0.0, 16), key(0.0, 16))
        self.assertEqual(key(-0.0, 32), key(0.0, 32))

    def test_scan_pair(self):
        rng = random.Random(811)
        for _ in range(25):
            values = [(rng.randrange(5), rng.randrange(5)) for _ in range(THREADS)]
            before, total = scan_pair(values)
            running = (0, 0)
            for v, p in zip(values, before):
                self.assertEqual(p, running)
                running = tuple(a + b for a, b in zip(running, v))
            self.assertEqual(total, running)

    def test_topk_shapes_masks_and_ties(self):
        rng = random.Random(512)
        for n in (1, 31, 32, 33, 300, 1023, 1024, 1025, 16384, 131072):
            scores = [bf16(rng.uniform(-3, 3)) if rng.randrange(4) else -math.inf for _ in range(n)]
            for k in sorted(set((1, min(n, 17), min(n, 512), n))):
                self.assertEqual(select(scores, k, offset=128), reference(scores, k, 128), (n, k))
        for scores in ([0.0, -0.0] * 513, [-math.inf] * 1025, [math.inf] * 1057,
                       [1.0] * 1023 + [2.0] * 259 + [1.0] * 511):
            self.assertEqual(select(scores, 512), reference(scores, 512))

    def test_prefix_across_256_tiles(self):
        scores = [1.0] * (256 * TILE + 19)
        scores[0], scores[-1] = 2.0, 3.0
        self.assertEqual(select(scores, 512), reference(scores, 512))

    def test_full_float_candidate_scores_and_partial_blocks(self):
        rng = random.Random(64)
        adjacent = [from_bits(0x3f800000 + i) for i in range(32)]
        for n in (1, 7, 8, 9, 31, 300, 1025, 16384):
            scores = [rng.choice(adjacent + [-math.inf, -0.0, 0.0, -1.0]) for _ in range(n)]
            for block in (1, 3, 8, 13):
                for k in (0, 1, 64, n):
                    self.assertEqual(candidates(scores, k, block), candidate_reference(scores, k, block),
                                     (n, block, k))
        for scores in ([-math.inf] * 300, [1.0] * 300, [math.inf] * 300):
            for k in (1, 5, 64):
                self.assertEqual(candidates(scores, k, 8), candidate_reference(scores, k, 8))

    def test_persistent_key_ownership(self):
        for n in (1, 300, 16384, 131072):
            parts = min((n + 7) // 8, 512)
            visits = [0] * n
            for part in range(parts):
                for warp in range(8):
                    for j in range(part * 8 + warp, n, parts * 8):
                        visits[j] += 1
            self.assertEqual(visits, [1] * n)


if __name__ == '__main__':
    unittest.main(verbosity=2)
