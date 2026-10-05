#!/usr/bin/env python3
"""CPU-only K5-12 storage/radix model; does not replace CUDA acceptance tests."""
from array import array
import hashlib
from pathlib import Path
import random
import struct

SEGMENT = 4096
COUNT_WORDS = 4
HEADER = 8
HISTOGRAM_WORDS = 256 * COUNT_WORDS
PARTITION_WORDS = 2 * COUNT_WORDS
RNG = random.Random(0x512)


def capacity(n):
    if n < SEGMENT:
        return False
    parts = (n - 1) // SEGMENT + 1
    available = n - HEADER
    return min(parts, 256) <= available // HISTOGRAM_WORDS and parts <= available // PARTITION_WORDS


def metadata_word(n, word):
    assert 0 <= word < n
    begin = word // SEGMENT * SEGMENT
    length = min(n - begin, SEGMENT)
    return 2 * begin + length + word % SEGMENT


def packed_word(j):
    return j // SEGMENT * SEGMENT + j


def key(bits):
    if bits & 0x7FFF == 0:
        bits = 0
    return bits ^ (0xFFFF if bits & 0x8000 else 0x8000)


def as_float(bits):
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


class Storage:
    def __init__(self, bits):
        self.n = len(bits)
        # Guard word pairs model a float-aligned, not necessarily 8-byte-aligned pointer.
        self.words = array('H', [0xA55A, 0x5AA5])
        for value in bits:
            self.words.extend((0, value))
        self.words.extend((0xDEAD, 0xBEEF))
        self.original = self.words[:]

    def get(self, word):
        assert 0 <= word < 2 * self.n
        return self.words[2 + word]

    def put(self, word, value):
        assert 0 <= word < 2 * self.n
        self.words[2 + word] = value

    def pack(self):
        segments = list(range(0, self.n, SEGMENT))
        RNG.shuffle(segments)
        for begin in segments:
            length = min(SEGMENT, self.n - begin)
            snapshot = [self.get(2 * (begin + i) + 1) for i in range(length)]
            order = list(range(length))
            RNG.shuffle(order)
            for i in order:
                self.put(2 * begin + i, snapshot[i])

    def bits(self, j):
        return self.get(packed_word(j))

    def metadata(self, word, value=None):
        address = metadata_word(self.n, word)
        if value is None:
            return self.get(address)
        self.put(address, value)

    def count(self, word, value=None):
        if value is None:
            return sum(self.metadata(word + r) << (16 * r) for r in range(COUNT_WORDS))
        assert 0 <= value < 1 << 64
        for r in range(COUNT_WORDS):
            self.metadata(word + r, value >> (16 * r) & 0xFFFF)

    def restore(self):
        segments = list(range(0, self.n, SEGMENT))
        RNG.shuffle(segments)
        for begin in segments:
            length = min(SEGMENT, self.n - begin)
            snapshot = [self.get(2 * begin + i) for i in range(length)]
            order = list(range(length))
            RNG.shuffle(order)
            for i in order:
                self.put(2 * (begin + i), 0)
                self.put(2 * (begin + i) + 1, snapshot[i])
        assert self.words == self.original, 'Restoration changed score bits or guards'


def select(storage, k, offset):
    n = storage.n
    k = min(max(k, 0), n)
    if k == 0:
        return []
    if k == n:
        return list(range(offset, offset + n))
    if not capacity(n):
        # Scratch-free branch: same two exact radix bytes; no storage mutation.
        keys = [key(storage.get(2 * j + 1)) for j in range(n)]
        prefix, mask, quota = 0, 0, k
        for shift in (8, 0):
            bins = [0] * 256
            for value in keys:
                if value & mask == prefix:
                    bins[value >> shift & 255] += 1
            for b in range(255, -1, -1):
                if bins[b] >= quota:
                    prefix |= b << shift
                    break
                quota -= bins[b]
            mask |= 255 << shift
        equal = 0
        result = []
        for j, value in enumerate(keys):
            if value > prefix or (value == prefix and equal < quota):
                result.append(j + offset)
            equal += value == prefix
        assert storage.words == storage.original
        return result

    storage.pack()
    parts = (n - 1) // SEGMENT + 1
    hist_parts = min(parts, 256)
    for shift in (8, 0):
        prefix = 0 if shift else storage.metadata(0)
        histograms = [[0] * 256 for _ in range(hist_parts)]
        for j in range(n):
            value = key(storage.bits(j))
            if shift or value & 0xFF00 == prefix:
                histograms[(j // 256) % hist_parts][value >> shift & 255] += 1
        for p, histogram in enumerate(histograms):
            for b, count in enumerate(histogram):
                storage.count(HEADER + (p * 256 + b) * COUNT_WORDS, count)
        quota = k if shift else storage.count(1)
        for b in range(255, -1, -1):
            count = sum(storage.count(HEADER + (p * 256 + b) * COUNT_WORDS)
                        for p in range(hist_parts))
            if count >= quota:
                storage.metadata(0, prefix | b << shift)
                storage.count(1, quota)
                break
            quota -= count

    cutoff, quota = storage.metadata(0), storage.count(1)
    for p in range(parts):
        values = [key(storage.bits(j)) for j in range(p * SEGMENT, min((p + 1) * SEGMENT, n))]
        storage.count(HEADER + p * PARTITION_WORDS, sum(v > cutoff for v in values))
        storage.count(HEADER + p * PARTITION_WORDS + COUNT_WORDS, sum(v == cutoff for v in values))
    greater, equal = 0, 0
    for p in range(parts):
        address = HEADER + p * PARTITION_WORDS
        g, e = storage.count(address), storage.count(address + COUNT_WORDS)
        storage.count(address, greater)
        storage.count(address + COUNT_WORDS, equal)
        greater, equal = greater + g, equal + e

    output = [None] * k
    order = list(range(parts))
    RNG.shuffle(order)
    for p in order:
        address = HEADER + p * PARTITION_WORDS
        greater, equal = storage.count(address), storage.count(address + COUNT_WORDS)
        for j in range(p * SEGMENT, min((p + 1) * SEGMENT, n)):
            value = key(storage.bits(j))
            if value > cutoff or (value == cutoff and equal < quota):
                address = greater + min(equal, quota)
                assert 0 <= address < k and output[address] is None
                output[address] = j + offset
            greater += value > cutoff
            equal += value == cutoff
    assert None not in output
    storage.restore()
    return output


def check_storage():
    # All 65536 BF16 encodings, including signed zero, infinities, and NaN payloads.
    shapes = [1, 2, 3, 255, 256, 257, 300, 4095, 4096, 4097, 4098,
              4111, 6143, 8191, 8192, 8193, 16384, 65536, 65537, 131071, 131072, 131073]
    for n in shapes:
        bits = [(j * 40503) & 0xFFFF for j in range(n)]
        storage = Storage(bits)
        storage.pack()
        for j, expected in enumerate(bits):
            assert storage.bits(j) == expected
        occupied = {packed_word(j) for j in range(n)}
        slack = {metadata_word(n, word) for word in range(n)}
        assert len(occupied) == len(slack) == n
        assert not occupied & slack and occupied | slack == set(range(2 * n))
        # Poison ALL slack, not merely the metadata footprint used by the selector.
        for word in range(n):
            storage.metadata(word, RNG.randrange(65536))
        for j, expected in enumerate(bits):
            assert storage.bits(j) == expected
        # Test actual four-halfword count serialization, including 64-bit extrema.
        for word in range(0, n - COUNT_WORDS + 1, max(4, (n // 16 // 4) * 4)):
            for value in (0, 1, 65535, 65536, 2**32 - 1, 2**32, 2**63, 2**64 - 1):
                storage.count(word, value)
                assert storage.count(word) == value
        storage.restore()
    return len(shapes)


def check_capacity():
    tested = 0
    for n in list(range(1, 1_050_000)) + [2**power + delta for power in range(21, 62)
                                       for delta in (-4097, -1, 0, 1, 4095)]:
        parts = (n - 1) // SEGMENT + 1
        required_words = HEADER + max(HISTOGRAM_WORDS * min(parts, 256), PARTITION_WORDS * parts)
        assert capacity(n) == (n >= SEGMENT and required_words <= n)
        if capacity(n):
            assert required_words * 2 <= n * 2
            last = required_words - 1
            address = metadata_word(n, last)
            begin = last // SEGMENT * SEGMENT
            length = min(n - begin, SEGMENT)
            assert 2 * begin + length <= address < 2 * (begin + length) <= 2 * n
        tested += 1
    return tested


def check_selection():
    tests = 0
    shapes = [1, 2, 3, 31, 32, 33, 255, 256, 257, 300, 1023, 1024, 4095,
              4096, 4097, 4111, 8191, 8192, 8193, 16384, 65537, 131072]
    palette = [0, 0x8000, 0xFF80, 0x7F80, 0x3F80, 0xBF80, 0x0080, 0x8080,
               1, 0x8001, 0x7F7F, 0xFF7F, 0x3F81, 0xBF81]
    for n in shapes:
        patterns = [[0xFF80] * n, [0x3F80] * n,
                    [palette[j % len(palette)] for j in range(n)],
                    [RNG.randrange(0x7F81) | (RNG.randrange(2) << 15) for _ in range(n)]]
        for bits in patterns:
            values = [as_float(b) for b in bits]
            ranked = sorted(range(n), key=lambda j: (-values[j], j))
            for k in sorted({-3, 0, 1, min(17, n), min(512, n), n // 2, n - 1, n, n + 3}):
                offset = -1209 if k % 2 else 8192
                want = [j + offset for j in sorted(ranked[:min(max(k, 0), n)])]
                got = select(Storage(bits), k, offset)
                assert got == want, (n, k, len(got), len(want))
                tests += 1
    # More than 256 partitions exercises grid-stride histogram and emission ownership.
    n = SEGMENT * 257 + 3
    bits = [palette[j % len(palette)] for j in range(n)]
    values = [as_float(b) for b in bits]
    ranked = sorted(range(n), key=lambda j: (-values[j], j))
    assert select(Storage(bits), 513, 37) == [j + 37 for j in sorted(ranked[:513])]
    return tests + 1


def check_unchanged_math():
    source = Path(__file__).resolve().parents[1].joinpath('k5_indexer.cu').read_text()
    # Exact source-region digests from K5-07 commit 466796657f. These regions
    # include score math, FP32 candidate maxima, and the scratch-free selector.
    regions = [
        ('__global__ void small_scores', '// Float-flip',
         '0ca5f132c99c7bbdde3e84cc92e667d892806fea3c6346e649ba706aba83f234'),
        ('__device__ __forceinline__ void load_key', '// Candidate block maxima',
         '87aa6ed058a298dc8faf0d02f2950d5fc79ae2283d8fde2b837d7dcec51f0505'),
        ('template <bool Candidate>\nstruct Values', '// Each segment owns exactly',
         '9d59838e033eb1bf44ff80b5ff00babade90cf993e445cdf91c34b0104f1609f'),
        ('void candidate_blocks', '}  // namespace strata::ds41::kernels',
         'ecb67ac6a6a41c90f36deff878ab67b3d22a24d5753452207a544ac9459e0839'),
    ]
    for start, end, expected in regions:
        begin = source.index(start)
        region = source[begin:source.index(end, begin)]
        assert hashlib.sha256(region.encode()).hexdigest() == expected, start
    return len(regions)


if __name__ == '__main__':
    print('Unchanged K5-07 source regions:', check_unchanged_math())
    print('Storage shapes:', check_storage(), '(all BF16 encodings, tails, poisoned slack, shuffled CTA/store order)')
    print('Capacity sizes:', check_capacity(), '(actual 2*t bytes, short fallback, large int64 counts)')
    print('Numerical cases:', check_selection(), '(exact two-byte cutoff, signed zero, ties, ascending emission)')
    print('PASS CPU model. GPU execution, graph capture, and performance are not tested.')
