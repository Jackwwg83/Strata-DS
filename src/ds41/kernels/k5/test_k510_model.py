#!/usr/bin/env python3
"""CPU models for K5-10 selection, tile addressing, and storage bounds.

These do not execute CUDA or establish GPU arithmetic parity/performance.
"""
import math
import random
import struct


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    if bits & 0x7f800000 != 0x7f800000:
        bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffffffff
    return struct.unpack('<f', struct.pack('<I', bits & 0xffff0000))[0]


def key(x, packed):
    bits = 0 if x == 0 else struct.unpack('<I', struct.pack('<f', x))[0]
    result = bits ^ (0xffffffff if bits & 0x80000000 else 0x80000000)
    return result >> 16 if packed else result


def select(values, k, packed=True, precomputed=False, offset=0):
    n = len(values)
    k = min(max(k, 0), n)
    if not k:
        return []
    if k == n:
        return list(range(offset, n + offset))
    keys = [key(x, packed) for x in values]
    first = 8 if packed else 24
    prefix = mask = 0
    quota = k
    histogram = [0] * 256
    if precomputed:
        assert packed and k >= 512
        for x in keys:
            histogram[x >> 8] += 1
        # Host-cleared output holds only 512 words and is loaded in full before
        # any output writes. Split 32-bit counters support int32 alignment.
        words = [word for count in histogram for word in (count & 0xffffffff, count >> 32)]
        assert len(words) <= k
        histogram = [words[2*i] | (words[2*i+1] << 32) for i in range(256)]
    for shift in range(first, -1, -8):
        if not (precomputed and shift == first):
            histogram = [0] * 256
            for x in keys:
                if x & mask == prefix:
                    histogram[x >> shift & 255] += 1
        for b in range(255, -1, -1):
            if histogram[b] >= quota:
                prefix |= b << shift
                break
            quota -= histogram[b]
        else:
            raise AssertionError('no threshold')
        mask |= 255 << shift
    out = [None] * k
    earlier_g = earlier_e = 0
    width, extra = divmod(n, 256)
    for tid in range(256):
        begin = width * tid + min(tid, extra)
        end = begin + width + (tid < extra)
        g = sum(x > prefix for x in keys[begin:end])
        e = sum(x == prefix for x in keys[begin:end])
        output = earlier_g + min(earlier_e, quota)
        local_e = earlier_e
        for j in range(begin, end):
            same = keys[j] == prefix
            chosen = keys[j] > prefix or (same and local_e < quota)
            local_e += same
            if chosen:
                assert 0 <= output < k and out[output] is None
                out[output] = j + offset
                output += 1
        earlier_g += g
        earlier_e += e
    assert None not in out
    return out


def reference(values, k, offset=0):
    k = min(max(k, 0), len(values))
    return [j + offset for j in sorted(sorted(range(len(values)), key=lambda j: (-values[j], j))[:k])]


def selection_tests():
    rng = random.Random(5010)
    cases = 0
    for n in (1, 2, 31, 127, 255, 256, 257, 300, 511, 512, 513, 1023, 1025,
              16384, 131072, 131201):
        populations = [[-math.inf] * n, [0.0] * n,
                       [rng.choice([-math.inf, -8.0, -0.0, 0.0, 0.25, 2.0, math.inf]) for _ in range(n)],
                       [bf16(rng.uniform(-8, 8)) for _ in range(n)]]
        for values in populations:
            for k in sorted(set((0, 1, 17, 511, 512, 513, n-1, n, n+7))):
                expected = reference(values, k, -31)
                actual_k = min(max(k, 0), n)
                assert select(values, k, offset=-31) == expected, (n, k)
                if 512 <= actual_k < n:
                    assert select(values, k, precomputed=True, offset=-31) == expected
                cases += 1
        # Full FP32 block scores, non-default block sizes, partially filled
        # final blocks, -inf removal and the required forced final block.
        for values in (populations[0], [f32(rng.uniform(-1, 1)) for _ in range(n)]):
            for block in (1, 3, 8, 17, n+7):
                maxima = [max(values[i:i+block]) for i in range(0, n, block)]
                maxima[-1] = math.inf
                for k in (0, 1, 64, len(maxima), len(maxima)+2):
                    got = select(maxima, k, packed=False)
                    want = reference(maxima, k)
                    got_set, want_set = set(got), set(want)
                    got_mask = [int(j//block in got_set and maxima[j//block] != -math.inf) for j in range(n)]
                    want_mask = [int(j//block in want_set and maxima[j//block] != -math.inf) for j in range(n)]
                    assert got_mask == want_mask, (n, block, k)
                    cases += 1
    return cases


def tile_tests():
    # All warp/fragment stores are disjoint and cover every head-key pair.
    writes = [[0] * 256 for _ in range(32)]
    for warp in range(8):
        for head_half in (0, 16):
            for key_half in (0, 16):
                for h in range(16):
                    for j in range(16):
                        writes[head_half+h][warp*32+key_half+j] += 1
    assert all(x == 1 for row in writes for x in row)
    # Four 32-dimension slices, alternating shared buffers; each prefetched
    # slice owns different addresses from the slice currently being consumed.
    buffers = [None, None]
    buffers[0] = [(j, d) for j in range(256) for d in range(32)]
    seen = set()
    for chunk in range(4):
        if chunk < 3:
            buffers[(chunk+1) & 1] = [(j, (chunk+1)*32+d) for j in range(256) for d in range(32)]
        for warp in range(8):
            for half in (0, 16):
                for step in (0, 16):
                    for j in range(16):
                        for d in range(16):
                            location = (warp*32+half+j)*32 + step+d
                            item = buffers[chunk & 1][location]
                            assert item == (warp*32+half+j, chunk*32+step+d)
                            assert item not in seen
                            seen.add(item)
    assert len(seen) == 256*128
    assert max(2*256*32*2, 32*256*4) + 32*128*2 + 32*4 + 256 == 41344
    for n in (513, 767, 1025, 16384, 65535, 65536, 65537, 131072, 131201):
        grid = min((n-1)//256+1, 256)
        visits = [0] * n
        for cta in range(grid):
            for base in range(cta*256, n, grid*256):
                for j in range(base, min(n, base+256)):
                    visits[j] += 1
                # All 16-byte copies are aligned, tails use valid first-row
                # source addresses, and the live mask controls zero fill.
                for vec in range(256*32//8):
                    j = base + vec//4
                    address = ((j if j<n else 0)*128 + (vec%4)*8)*2
                    assert address % 16 == 0
                    assert address+16 <= n*128*2
        assert all(x == 1 for x in visits)
    # Dyadic, exactly accumulated dots test the required three BF16 rounding
    # stages, mask epilogue and increasing-head sum without order ambiguity.
    q = [[((h*3+d*5)%13-6)/8 for d in range(128)] for h in range(32)]
    weights = [bf16((h%5+1)/64 * (-1 if h%3 == 0 else 1)) for h in range(32)]
    for j in range(257):
        row = [((j*7+d*3)%17-8)/8 for d in range(128)]
        total = ref = 0.0
        for h in range(32):
            partial = 0.0
            for chunk in range(4):
                for step in (0, 16):
                    for d in range(chunk*32+step, chunk*32+step+16):
                        partial = f32(partial + q[h][d]*row[d])
            direct = sum(q[h][d]*row[d] for d in range(128))
            assert partial == direct
            total = f32(total + bf16(f32(max(bf16(partial), 0.0)*weights[h])))
            ref = f32(ref + bf16(f32(max(bf16(direct), 0.0)*weights[h])))
        live = j%4 != 0
        assert (bf16(total) if live else -math.inf) == (bf16(ref) if live else -math.inf)


def counter_tests():
    rng = random.Random(10)
    for initial in (0, 2**32-32, 2**32-1, 2**32, 2**40+2**32-10):
        lo, hi = initial & 0xffffffff, initial >> 32
        exact = initial
        for _ in range(128):
            add = rng.randrange(1, 33)
            old = lo
            lo = (lo + add) & 0xffffffff
            hi += old > 0xffffffff-add
            exact += add
            assert lo | (hi << 32) == exact
    values = []
    for packed in range(65536):
        value = struct.unpack('<f', struct.pack('<I', packed << 16))[0]
        if not math.isnan(value):
            values.append(value)
    values.sort()
    keys = [key(x, True) for x in values]
    assert keys == sorted(keys)
    assert key(-0.0, True) == key(0.0, True)


if __name__ == '__main__':
    count = selection_tests()
    tile_tests()
    counter_tests()
    print(f'PASS {count} selection/candidate cases; tile mapping, rounding, tails, storage and 64-bit carries (CPU models only)')
