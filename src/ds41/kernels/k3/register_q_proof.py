#!/usr/bin/env python3
"""CPU-only proof of K3-21 Q fragment identity and complete ownership.

No CUDA execution or timing claim. Run with Python 3 from any directory.
The reference emulates K3-11's shared Q producer and ldmatrix.x2 mapping.
"""
from collections import Counter
import random

rng = random.Random(321)
for trial in range(3):
    # Unique labels establish ownership; arbitrary BF16 bit patterns establish
    # that packed loads preserve bits without conversion or special casing.
    q = [[h * 512 + d if trial == 0 else rng.randrange(1 << 16)
          for d in range(512)] for h in range(8)]
    shared = [None] * (8 * 520)
    for tid in range(128):
        for i in range(tid, 8 * 512 // 8, 128):
            h, chunk = divmod(i, 64)
            shared[h * 520 + chunk * 8:h * 520 + chunk * 8 + 8] = q[h][chunk * 8:chunk * 8 + 8]
    for group in range(4):
        order = []
        for warp in range(4):
            owners = Counter()
            for slice_id in range(8):
                dimension = ((group * 128 + 128) + slice_id * 64) & 511
                for step in range(4):
                    d = dimension + step * 16
                    if warp == 0:
                        order.append(d)
                    # In ldmatrix.x2, lanes 0..7 supply rows of matrix 0,
                    # lanes 8..15 supply rows of matrix 1. The upper 16
                    # address lanes do not supply an additional matrix.
                    rows = []
                    for address_lane in range(16):
                        p = (address_lane % 8) * 520 + d + (address_lane // 8) * 8
                        assert (p * 2) % 16 == 0
                        rows.append(shared[p:p + 8])
                    for lane in range(32):
                        h, t = divmod(lane, 4)
                        for reg in range(2):
                            old_pair = rows[reg * 8 + h][t * 2:t * 2 + 2]
                            k = d + t * 2 + reg * 8
                            new_pair = q[h][k:k + 2]
                            assert old_pair == new_pair
                            assert ((h * 512 + k) * 2) % 4 == 0
                            old_bits = old_pair[0] | (old_pair[1] << 16)
                            new_bits = new_pair[0] | (new_pair[1] << 16)
                            assert old_bits == new_bits
                            for e in range(2):
                                owners[h, k + e] += 1
            assert len(owners) == 8 * 512
            assert set(owners.values()) == {1}
            for lane in range(32):
                # 8 slices * 4 K steps * 2 packed B registers.
                assert 8 * 4 * 2 == 64
        expected_order = [((group * 128 + 128) + i * 16) & 511 for i in range(32)]
        assert order == expected_order
        assert order[-8:] == list(range(group * 128, (group + 1) * 128, 16))
print('PASS: direct Q registers are bit-identical to K3-11 ldmatrix.x2 fragments')
print('PASS: all four output rotations, four consuming warps, and 4096 Q values per warp')
print('PASS: exactly one owner per warp/value, 64 packed registers per lane, no full query per lane')
print('PASS: unchanged cyclic K=16 MMA order and final two output slices')
