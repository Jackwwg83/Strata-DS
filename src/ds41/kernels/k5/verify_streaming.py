#!/usr/bin/env python3
"""Dependency-free CPU model for K5's chunk ownership and exact selection.

This is not a CUDA execution test. Run with python3 from any directory.
"""
import math
import random
import struct

CHUNK, CAP = 16384, 512


def bits(x):
    return struct.unpack('I', struct.pack('f', x))[0]


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def bf16(x):
    u = bits(x)
    u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000
    return struct.unpack('f', struct.pack('I', u))[0]


def key(x):
    u = 0 if x == 0 else bits(x)
    return u ^ (0xffffffff if u & 0x80000000 else 0x80000000)


def cutoff(keys, k, shifts):
    prefix, remaining = 0, k
    for shift in shifts:
        mask = (0xffffffff << (shift + 8)) & 0xffffffff
        bins = [0] * 256
        for x in keys:
            if x & mask == prefix:
                bins[(x >> shift) & 255] += 1
        for b in range(255, -1, -1):
            if bins[b] >= remaining:
                prefix |= b << shift
                break
            remaining -= bins[b]
    return prefix, remaining


def bitonic(a):
    size = 2
    while size <= len(a):
        stride = size >> 1
        while stride:
            for i in range(len(a)):
                j = i ^ stride
                if j > i and ((not i & size and a[i] > a[j]) or (i & size and a[i] < a[j])):
                    a[i], a[j] = a[j], a[i]
            stride >>= 1
        size <<= 1
    return a


def reference(scores, k, offset=0):
    return sorted(sorted(range(len(scores)), key=lambda j: (-scores[j], j))[:max(0, k)]) if not offset else [j + offset for j in reference(scores, k)]


def fallback(scores, k, offset=0, full_float=False, drop_inf=False):
    k = max(0, min(len(scores), k))
    kk = [key(x) for x in scores]
    threshold, needed = (cutoff(kk, k, (24, 16, 8, 0) if full_float else (24, 16))
                         if 0 < k < len(scores) else (0, k))
    seen, out = 0, []
    for i, x in enumerate(kk):
        x = x if full_float else x & 0xffff0000
        eq = x == threshold
        chosen = k > 0 and (k >= len(scores) or x > threshold or (eq and seen < needed))
        seen += eq
        if chosen and (not drop_inf or scores[i] != -math.inf):
            out.append(i + offset)
    return out


def stream(scores, k, offset, rng):
    k = max(0, min(k, len(scores)))
    if not k:
        return []
    if k == len(scores):
        return [i + offset for i in range(k)]
    if k > CAP:
        return fallback(scores, k, offset)
    # The caller owns exactly k slots. None represents unwritten device bytes.
    out = [None] * k
    for begin in range(0, len(scores), CHUNK):
        values = scores[begin:begin + CHUNK]
        take = min(k, len(values))
        carried = min(k, begin)
        old = out[:carried]
        assert all(p is not None and 0 <= p < begin for p in old)
        # Every read of the preceding output precedes every write this phase.
        snapshot = list(old)
        local_keys = [key(x) >> 16 for x in values]
        threshold, needed = cutoff(local_keys, take, (8, 0))
        above_groups, equal = [], []
        for start in range(0, len(values), 32):
            above_groups.append([start + lane for lane in range(min(32, len(values) - start))
                                 if local_keys[start + lane] > threshold])
        rng.shuffle(above_groups)  # arbitrary warp atomic reservation order
        above = [p for group in above_groups for p in group]
        for i, x in enumerate(local_keys):
            if x == threshold and len(equal) < needed:
                equal.append(i)
        assert len(above) == take - needed
        local = [begin + p for p in above + equal]
        ranks = [(0xffff - (key(scores[p]) >> 16)) << 32 | p for p in local]
        ranks += [(1 << 64) - 1] * (CAP - len(ranks))
        ranks += [(0xffff - (key(scores[p]) >> 16)) << 32 | p for p in snapshot]
        ranks += [(1 << 64) - 1] * (2 * CAP - len(ranks))
        bitonic(ranks)
        written = min(k, begin + len(values))
        for i in range(written):
            out[i] = ranks[i] & 0xffffffff
        assert sorted(out[:written]) == reference(scores[:begin + len(values)], k)
    ids = bitonic(out + [0xffffffff] * (CAP - k))
    return [p + offset for p in ids[:k]]


def candidates(scores, k, block):
    if not scores:
        return []
    maxima = [max(scores[i:i + block]) for i in range(0, len(scores), block)]
    maxima[-1] = math.inf
    selected = set(fallback(maxima, k, full_float=True, drop_inf=True))
    out = [int(i // block in selected) for i in range(len(scores))]
    want = set(reference(maxima, k))
    want = {i for i in want if maxima[i] != -math.inf}
    assert out == [int(i // block in want) for i in range(len(scores))]
    return out


def main():
    rng = random.Random(508)
    runs = 0
    # Every fast-path phase is checked against the full prefix oracle, not just
    # the final output. Ragged chunks, tie boundaries, mask starvation, zero
    # signs, positive infinity and negative infinity are all represented.
    for n in (1, 31, 300, 513, 1025, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK + 257, 131072):
        distributions = (
            [bf16(rng.uniform(-20, 20)) for _ in range(n)],
            [-math.inf if rng.randrange(4) else bf16(rng.randrange(-3, 4)) for _ in range(n)],
            [-math.inf] * n,
            [(-0.0, 0.0, math.inf, -math.inf)[i % 4] for i in range(n)],
        )
        for scores in distributions:
            for k in (1, 511, 512, 513, n, n + 1):
                offset = -19 if runs % 2 else 1031
                got = stream(scores, k, offset, rng)
                assert got == reference(scores, k, offset), (n, k)
                runs += 1
    for n in (1, 7, 31, 300, 1025, 16385):
        scores = [f32(rng.uniform(-20, 20)) for _ in range(n)]
        for b in (1, 3, 8, 47, n + 5):
            for k in (0, 1, 3, 64, n + 1):
                candidates(scores, k, b)
                candidates([-math.inf] * n, k, b)
                candidates([math.inf if i % 3 == 0 else -0.0 for i in range(n)], k, b)
                runs += 3
    for n in (0, 1, 513, 33000):
        scores = [bf16(rng.randrange(-2, 3)) for _ in range(n)]
        for k in (0, 1, 513, 1025, max(1, n - 1), n + 1):
            assert fallback(scores, k) == reference(scores, k)
            runs += 1
    print(f'PASS: {runs} selection cases; prefix ownership and every chunk merge checked')
    print('GPU execution, numerical parity and timings still require the queue test')


if __name__ == '__main__':
    main()
