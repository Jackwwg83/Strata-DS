#!/usr/bin/env python3
"""CPU-only K5-13 packed ownership, radix, restoration and tensor source model."""
from array import array
import hashlib
from pathlib import Path
import random
import re
import struct

SEGMENT, HEADER, COUNT_WORDS = 2048, 8, 4
RNG = random.Random(0x513)


def count(n):
    return n // SEGMENT


def length(n, p):
    assert 0 <= p < count(n)
    return n - p * SEGMENT if p + 1 == count(n) else SEGMENT


def packed_word(n, j):
    p = min(j // SEGMENT, count(n) - 1)
    return p * SEGMENT + j


def metadata_word(n, word):
    p, offset = divmod(word, SEGMENT)
    assert 0 <= offset < length(n, p)
    return 2 * p * SEGMENT + length(n, p) + offset


def key(bits):
    if bits & 0x7fff == 0:
        bits = 0
    return bits ^ (0xffff if bits & 0x8000 else 0x8000)


def value(bits):
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


class Storage:
    def __init__(self, bits):
        self.n, self.original = len(bits), list(bits)
        # One float of guard on either side, also model four-byte base alignment.
        self.words = array('H', [0xa55a, 0x5aa5] + [0xcccc] * (2 * self.n) + [0xdead, 0xbeef])

    def get(self, word):
        assert 0 <= word < 2 * self.n
        return self.words[2 + word]

    def put(self, word, data):
        assert 0 <= word < 2 * self.n
        self.words[2 + word] = data

    def bits(self, j):
        return self.get(packed_word(self.n, j))

    def metadata(self, word, data=None):
        address = metadata_word(self.n, word)
        if data is None:
            return self.get(address)
        self.put(address, data)

    def count(self, word, data=None):
        if data is None:
            return sum(self.metadata(word + i) << (16 * i) for i in range(COUNT_WORDS))
        assert 0 <= data < 2**64
        for i in range(COUNT_WORDS):
            self.metadata(word + i, data >> (16 * i) & 0xffff)

    def fused(self):
        # Arbitrary CTA completion order, including owner loops above 256 parts.
        owners = list(range(count(self.n)))
        RNG.shuffle(owners)
        for p in owners:
            begin, end = p * SEGMENT, p * SEGMENT + length(self.n, p)
            histogram = [0] * 256
            for base in range(begin, end, 64):
                order = list(range(base, min(end, base + 64)))
                RNG.shuffle(order)
                for j in order:
                    # Store the produced BF16 directly, never read float scores.
                    self.put(begin + j, self.original[j])
                    histogram[key(self.original[j]) >> 8] += 1
            assert sum(histogram) == end - begin
            for b, hits in enumerate(histogram):
                self.count(p * SEGMENT + HEADER + b * COUNT_WORDS, hits)

    def restore(self):
        owners = list(range(count(self.n)))
        RNG.shuffle(owners)
        for p in owners:
            begin, size = p * SEGMENT, length(self.n, p)
            snapshot = [self.get(2 * begin + i) for i in range(size)]
            order = list(range(size))
            RNG.shuffle(order)
            for i in order:
                self.put(2 * (begin + i), 0)
                self.put(2 * (begin + i) + 1, snapshot[i])
        assert self.words[:2] == array('H', [0xa55a, 0x5aa5])
        assert self.words[-2:] == array('H', [0xdead, 0xbeef])
        for j, bits in enumerate(self.original):
            assert self.get(2 * j) == 0 and self.get(2 * j + 1) == bits


def select(bits, k, offset=0):
    n = len(bits)
    k = min(max(k, 0), n)
    if k == 0:
        return []
    if k == n:
        return list(range(offset, offset + n))
    if n < SEGMENT:
        # Independently tested four/two-byte shared-only selector.
        return [j + offset for j in sorted(sorted(range(n), key=lambda j: (-value(bits[j]), j))[:k])]
    storage = Storage(bits)
    storage.fused()
    for shift in (8, 0):
        prefix = 0 if shift else storage.metadata(0)
        if not shift:
            for p in range(count(n)):
                bins = [0] * 256
                for j in range(p * SEGMENT, p * SEGMENT + length(n, p)):
                    v = key(storage.bits(j))
                    if v & 0xff00 == prefix:
                        bins[v & 255] += 1
                for b, hits in enumerate(bins):
                    storage.count(p * SEGMENT + HEADER + b * COUNT_WORDS, hits)
        remaining = k if shift else storage.count(1)
        for b in range(255, -1, -1):
            hits = sum(storage.count(p * SEGMENT + HEADER + b * COUNT_WORDS) for p in range(count(n)))
            if hits >= remaining:
                storage.metadata(0, prefix | b << shift)
                storage.count(1, remaining)
                break
            remaining -= hits
    cutoff, quota = storage.metadata(0), storage.count(1)
    for p in range(count(n)):
        keys = [key(storage.bits(j)) for j in range(p * SEGMENT, p * SEGMENT + length(n, p))]
        storage.count(p * SEGMENT + HEADER, sum(v > cutoff for v in keys))
        storage.count(p * SEGMENT + HEADER + COUNT_WORDS, sum(v == cutoff for v in keys))
    g, e = 0, 0
    for p in range(count(n)):
        address = p * SEGMENT + HEADER
        own_g, own_e = storage.count(address), storage.count(address + COUNT_WORDS)
        storage.count(address, g)
        storage.count(address + COUNT_WORDS, e)
        g, e = g + own_g, e + own_e
    result = [None] * k
    owners = list(range(count(n)))
    RNG.shuffle(owners)
    for p in owners:
        address = p * SEGMENT + HEADER
        g, e = storage.count(address), storage.count(address + COUNT_WORDS)
        begin, end = p * SEGMENT, p * SEGMENT + length(n, p)
        # Model each padded 1024-position emission batch's valid end explicitly.
        for base in range(begin, end, 1024):
            for j in range(base, base + 1024):
                if j >= end:
                    continue
                v = key(storage.bits(j))
                if v > cutoff or (v == cutoff and e < quota):
                    pos = g + min(e, quota)
                    assert 0 <= pos < k and result[pos] is None
                    result[pos] = j + offset
                g += v > cutoff
                e += v == cutoff
    assert None not in result
    storage.restore()
    return result


def test_storage():
    shapes = [2048, 2049, 2050, 2111, 3071, 4095, 4096, 4097, 4111, 6143, 6144,
              8191, 8192, 8193, 16384, 65536, 65537, 131071, 131072, 131073]
    for n in shapes:
        bits = [(j * 40503) & 0xffff for j in range(n)]
        storage = Storage(bits)
        storage.fused()
        writes = set()
        for p in range(count(n)):
            begin, size = p * SEGMENT, length(n, p)
            score_words = set(range(2 * begin, 2 * begin + size))
            spare_words = set(range(2 * begin + size, 2 * (begin + size)))
            assert not score_words & spare_words
            assert not writes & (score_words | spare_words)
            writes |= score_words | spare_words
            # Poison the entire spare half, including unused appended tail.
            for word in spare_words:
                storage.put(word, RNG.randrange(65536))
            for data in (0, 1, 65535, 65536, 2**32, 2**63, 2**64 - 1):
                storage.count(p * SEGMENT + HEADER, data)
                assert storage.count(p * SEGMENT + HEADER) == data
        assert writes == set(range(2 * n))
        assert [storage.bits(j) for j in range(n)] == bits
        storage.restore()
    return len(shapes)


def test_capacity():
    cases = 0
    for n in list(range(2048, 1_050_000)) + [2**p + d for p in range(21, 60) for d in (-2049, -1, 0, 1, 2047)]:
        parts = count(n)
        for p in {0, parts - 1}:
            size = length(n, p)
            assert SEGMENT <= size < 2 * SEGMENT
            begin = p * SEGMENT
            assert 0 <= begin < begin + size <= n
            for word in (0, 4, HEADER, HEADER + 1024 - 1):
                address = metadata_word(n, p * SEGMENT + word)
                assert 2 * begin + size <= address < 2 * (begin + size)
        assert (parts - 1) * SEGMENT + length(n, parts - 1) == n
        cases += 1
    return cases


def test_selection():
    shapes = [2048, 2049, 2055, 2111, 3071, 4095, 4096, 4097, 6143, 8191, 8192, 16384, 65537, 131072]
    palette = [0, 0x8000, 0xff80, 0x7f80, 0x3f80, 0xbf80, 1, 0x8001, 0x7f7f, 0xff7f, 0x3f81, 0xbf81]
    tests = 0
    for n in shapes:
        patterns = [[0xff80] * n, [0x3f80] * n, [palette[j % len(palette)] for j in range(n)],
                    [RNG.randrange(0x7f81) | RNG.randrange(2) << 15 for _ in range(n)]]
        for bits in patterns:
            ranked = sorted(range(n), key=lambda j: (-value(bits[j]), j))
            for k in (-3, 0, 1, 17, 512, n // 2, n - 1, n, n + 3):
                offset = -1209 if k % 2 else 8192
                want = [j + offset for j in sorted(ranked[:min(max(k, 0), n)])]
                assert select(bits, k, offset) == want, (n, k)
                tests += 1
    n = 257 * SEGMENT + 2047
    bits = [palette[j % len(palette)] for j in range(n)]
    ranked = sorted(range(n), key=lambda j: (-value(bits[j]), j))
    assert select(bits, 513, 37) == [j + 37 for j in sorted(ranked[:513])]
    return tests + 1


def test_source():
    source = Path(__file__).resolve().parents[1].joinpath('k5_indexer.cu').read_text()
    # Exact K5-07/K5-12 full-FP32 candidate and small scorer source provenance.
    regions = [('__global__ void small_scores', '// Float-flip',
                '0ca5f132c99c7bbdde3e84cc92e667d892806fea3c6346e649ba706aba83f234'),
               ('template <bool Candidate>\nstruct Values', '// The final short tail',
                '9d59838e033eb1bf44ff80b5ff00babade90cf993e445cdf91c34b0104f1609f'),
               ('void candidate_blocks', '}  // namespace strata::ds41::kernels',
                'ecb67ac6a6a41c90f36deff878ab67b3d22a24d5753452207a544ac9459e0839')]
    for start, end, expected in regions:
        a = source.index(start)
        assert hashlib.sha256(source[a:source.index(end, a)].encode()).hexdigest() == expected, start
    for token in ('cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaStreamSynchronize',
                  'cudaDeviceSynchronize', 'cudaStreamIsCapturing', 'cudaEvent', 'thread_local', 'pack_scores'):
        assert token not in source, token
    assert all(launch.strip().endswith(', stream') for launch in re.findall(r'<<<(.*?)>>>', source))
    # WMMA sequence and rounded head reduction are unchanged, modulo indentation.
    tensor_start = source.index('namespace wm = nvcuda::wmma;')
    tensor_end = source.index('__syncthreads();', tensor_start)
    tensor = re.sub(r'\s+', '', source[tensor_start:tensor_end])
    assert hashlib.sha256(tensor.encode()).hexdigest() == '7b2c2e5098b67464f95cf968dcd4a9472b17f0b1caffc1bde11b3e736eb095f3'
    assert 'total+=rounded(fmaxf(rounded(dots[h*64+tid]),0.0f)*weights[h]);' in re.sub(r'\s+', '', source)
    assert 'score = rounded(total);' in source
    writes = [0] * (32 * 64)
    for warp in range(4):
        for half in range(2):
            for row in range(16):
                for col in range(16):
                    writes[(half * 16 + row) * 64 + warp * 16 + col] += 1
        for d in range(0, 128, 16):
            assert d * 2 % 32 == 0
            assert (warp * 16 * 128 + d) * 2 % 32 == 0
    assert all(x == 1 for x in writes)
    return len(regions)


if __name__ == '__main__':
    print('PASS source provenance/tensor tile/stream-only checks:', test_source(), flush=True)
    print('PASS inter-CTA storage/restoration shapes:', test_storage(), flush=True)
    print('PASS metadata capacity and merged-tail sizes:', test_capacity(), flush=True)
    print('PASS exact fused/packed radix selection cases:', test_selection(), flush=True)
