#!/usr/bin/env python3
"""CPU model of K5-11's caller-storage radix stages.

This checks integer addressing, phase contents, tie selection, exact restoration,
and BF16 score representation. It does not execute CUDA instructions or model
WMMA accumulation order; GPU score parity must be established by the fixed test.
"""
import math
import random
import struct
from test_selection_model import reference, select, key


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def value(x):
    return struct.unpack('<f', struct.pack('<I', x))[0]


def bf(x):
    b = bits(f32(x))
    return value(((b + 0x7fff + ((b >> 16) & 1)) >> 16) << 16)


def parallel_model(values, k):
    n = len(values)
    if n < 4096 or k == 0 or k == n:
        return select(values, k, True)
    original = [bits(x) for x in values]
    assert all(x & 65535 == 0 for x in original)
    words = original.copy()
    parts, hist_parts = (n + 4095) // 4096, min((n + 4095) // 4096, 256)
    written = 8 + max(1024 * hist_parts, 8 * parts)
    assert written <= n

    def high_key(j):
        b = words[j] >> 16
        b = 0 if b & 0x7fff == 0 else b
        return b ^ (65535 if b & 0x8000 else 32768)

    def store(j, v):
        assert 0 <= j < n and 0 <= v < 65536
        words[j] = words[j] & 0xffff0000 | v

    def load(j):
        assert 0 <= j < n
        return words[j] & 65535

    def store_count(j, v):
        for r in range(4):
            store(j + r, v >> (r * 16) & 65535)

    def load_count(j):
        return sum(load(j + r) << (r * 16) for r in range(4))

    prefix, remaining = 0, k
    for shift in (8, 0):
        hist = [[0] * 256 for _ in range(hist_parts)]
        for j in range(n):
            q = high_key(j)
            if shift == 8 or q & 0xff00 == prefix:
                # CUDA CTA ownership is grid-strided chunks of 256 positions.
                hist[j // 256 % hist_parts][q >> shift & 255] += 1
        for p, counts in enumerate(hist):
            for b, count in enumerate(counts):
                store_count(8 + (p * 256 + b) * 4, count)
        bins = [sum(load_count(8 + (p * 256 + b) * 4) for p in range(hist_parts))
                for b in range(256)]
        for b in range(255, -1, -1):
            if bins[b] >= remaining:
                prefix |= b << shift
                break
            remaining -= bins[b]
        store(0, prefix)
        store_count(1, remaining)
        assert all(words[j] >> 16 == original[j] >> 16 for j in range(n))

    # Reuse histogram metadata only after the second chooser has completed.
    for p in range(parts):
        keys = [high_key(j) for j in range(p * 4096, min(n, (p + 1) * 4096))]
        store_count(8 + p * 8, sum(q > prefix for q in keys))
        store_count(12 + p * 8, sum(q == prefix for q in keys))
    g, e = 0, 0
    for p in range(parts):
        own_g, own_e = load_count(8 + p * 8), load_count(12 + p * 8)
        store_count(8 + p * 8, g)
        store_count(12 + p * 8, e)
        g, e = g + own_g, e + own_e
    assert g < k <= g + e

    out = [None] * k
    for p in range(parts):
        g, e = load_count(8 + p * 8), load_count(12 + p * 8)
        for j in range(p * 4096, min(n, (p + 1) * 4096)):
            q = high_key(j)
            if q > prefix or (q == prefix and e < remaining):
                destination = g + min(e, remaining)
                assert out[destination] is None
                out[destination] = j
            g += q > prefix
            e += q == prefix
    for j in range(written):
        store(j, 0)
    assert words == original, 'every score bit must be restored'
    return out


def main():
    rng = random.Random(5007)
    cases = 0
    populations = [-math.inf, -8., -0., 0., 0.25, 1., 2., math.inf]
    for n in (1, 31, 300, 512, 513, 4095, 4096, 4097, 8191, 8192,
              16384, 131072, 131201, 262273, 1048577):
        for values in ([rng.choice(populations) for _ in range(n)], [-math.inf] * n):
            for k in sorted(set((0, 1, min(17, n), min(512, n), n // 2, n))):
                assert parallel_model(values, k) == reference(values, k), (n, k)
                cases += 1
    # Capacity and 64-bit count roundtrips beyond the acceptance-test sizes.
    for n in (4096, 4097, 2**24 + 1, 2**32 + 7, 2**40 - 1):
        p = (n + 4095) // 4096
        assert 8 + max(1024 * min(p, 256), 8 * p) <= n
    for count in (0, 65535, 65536, 2**32 - 1, 2**32, 2**40):
        packed = [(count >> (r * 16)) & 65535 for r in range(4)]
        assert sum(v << (16 * r) for r, v in enumerate(packed)) == count
    # Restoration itself also preserves upper NaN payloads, infinities, signed
    # zeros, and subnormal encodings independently of comparator semantics.
    for high in range(65536):
        original = high << 16
        changed = original | rng.randrange(65536)
        assert changed & 0xffff0000 == original
    print(f'PASS {cases} caller-storage selections; '
          'all 65536 BF16 restoration encodings; capacity and 64-bit counts')


if __name__ == '__main__':
    main()
