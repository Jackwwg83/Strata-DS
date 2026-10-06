#!/usr/bin/env python3
"""CPU structural checks for K2-14; no CUDA execution or timing prediction."""
from collections import Counter
import validate_grouped as control


def address(groups, row, k):
    return ((row * 32 + k) ^ ((row & 7) * 8)) if groups > 1 else row * 40 + k


def producers(tm, tn, groups, is_a):
    for tid in range(tn * 2):
        if is_a:
            if tid < tm * 4:
                for group in range(groups):
                    yield tid, group, group * tm + tid // 4, (tid % 4) * 8, 8
        elif groups > 1:
            for packet in range(2):
                yield tid, packet, tid // 4 + packet * (tn // 2), (tid % 4) * 8, 8
        else:
            yield tid, 0, tid // 2, (tid % 2) * 16, 16


def consumers(tm, tn, groups, is_a):
    for tid in range(tn * 2):
        warp, lane = divmod(tid, 32)
        wm, wn = (warp // (tn // tm)) * 16, (warp % (tn // tm)) * tm
        for k in (0, 16):
            if is_a:
                for group in range(groups):
                    yield group * tm + wm + lane % 16, k + (lane // 16) * 8
            else:
                for j in range(tm // 8):
                    yield wn + j * 8 + lane % 8, k + ((lane // 8) % 2) * 8


def check_publication():
    vectors = consumer_vectors = tail_stages = 0
    for tm, tn, groups in control.SHAPES:
        stride = 32 if groups > 1 else 40
        assert (groups * tm + tn) * stride * 2 <= 99 * 1024
        for is_a, rows in ((True, groups * tm), (False, tn)):
            owners = list(producers(tm, tn, groups, is_a))
            logical = Counter((r, k + i) for _, _, r, k, count in owners for i in range(count))
            assert logical == Counter({(r, k): 1 for r in range(rows) for k in range(32)})
            physical = [address(groups, r, k) for r, k in logical]
            assert len(set(physical)) == rows * 32
            assert min(physical) == 0 and max(physical) < rows * stride
            if groups > 1:
                assert set(physical) == set(range(rows * 32))
            # Grouped B and every A use 8-BF16, 16-byte publications. Narrow B
            # preserves its old 16-FP8 packet, decoded to scalar BF16 pairs.
            for _, _, r, k, count in owners:
                for sub in range(0, count, 8):
                    start = address(groups, r, k + sub)
                    assert start % 8 == 0
                    assert [address(groups, r, k + sub + i) for i in range(8)] == list(range(start, start + 8))
                    vectors += 1
            for valid_rows in range(rows + 1):
                shared = [None] * (rows * stride)
                for _, _, r, k, count in owners:
                    for i in range(count):
                        dest = address(groups, r, k + i)
                        assert shared[dest] is None
                        shared[dest] = (r, k + i) if r < valid_rows else 0
                # ldmatrix's eight adjacent BF16 elements must retrieve exactly
                # the control's logical fragment for every lane and K half.
                for r, k in consumers(tm, tn, groups, is_a):
                    start = address(groups, r, k)
                    assert start % 8 == 0 and start + 7 < len(shared)
                    expected = [(r, k + i) if r < valid_rows else 0 for i in range(8)]
                    assert shared[start:start + 8] == expected
                    consumer_vectors += 1
                tail_stages += 1
    print(f'PASS ownership, allocation, vector alignment/continuity and consumer fragments: '
          f'{vectors} producer vectors, {consumer_vectors} consumer vectors, {tail_stages} tail stages')


def check_global_and_scales():
    for tm, tn, groups in control.SHAPES:
        owners = list(producers(tm, tn, groups, False))
        rounds = 2 if groups > 1 else 1
        width = 8 if groups > 1 else 16
        for K in (32, 64, 96, 1280, 5120, 8192):
            for kb in {0, K - 32}:
                for _, _, row, k, count in owners:
                    index = row * K + kb + k
                    assert count == width and index % width == 0
                    assert kb + k + count <= K
                    assert index + count <= tn * K
        # N need not be divisible by 32. Check every final-tile length and
        # multiple tile bases, proving valid packet lanes use lane zero's
        # same scale and that no invalid scale byte is loaded.
        for col_base in (0, tn, 7 * tn):
            for valid_n in range(tn + 1):
                N = col_base + valid_n
                for packet in range(rounds):
                    for warp in range(tn * 2 // 32):
                        rows = [row for tid, q, row, _, _ in owners if tid // 32 == warp and q == packet]
                        assert len(rows) == 32 and rows[0] == min(rows)
                        first = col_base + rows[0]
                        for row in rows:
                            col = col_base + row
                            if col < N:
                                assert first < N
                                assert col // 32 == first // 32
                                assert first // 32 < (N + 31) // 32
    print('PASS global vector bounds/alignment, scale ownership and every N tail for all specializations')


def check_bank_model():
    def rounds(starts, words):
        result = 0
        for first in range(0, 32, 32 // words):
            payload = [starts[t] // 2 + i for t in range(first, first + 32 // words) for i in range(words)]
            assert len(set(payload)) == len(payload)
            result += max(Counter(x % 32 for x in payload).values())
        return result
    a_old = sum(rounds([(g * 64 + t // 4) * 40 + t % 4 * 8 for t in range(32)], 4) for g in range(2))
    b_old = sum(rounds([t // 2 * 40 + t % 2 * 16 + 2 * pair for t in range(32)], 1) for pair in range(8))
    a_new = sum(rounds([address(2, g * 64 + t // 4, t % 4 * 8) for t in range(32)], 4) for g in range(2))
    b_new = sum(rounds([address(2, t // 4 + q * 64, t % 4 * 8) for t in range(32)], 4) for q in range(2))
    assert (a_old, b_old, a_new, b_new) == (16, 32, 8, 8)
    for row0 in range(0, 128, 8):
        for k in (0, 8, 16, 24):
            words = [address(2, row, k) // 2 + i for row in range(row0, row0 + 8) for i in range(4)]
            assert len(set(words)) == 32 and len({w % 32 for w in words}) == 32
    # Negative control: XORing only K is not a bijection and exceeds capacity.
    bad = [r * 32 + (k ^ ((r & 7) * 8)) for r in range(128) for k in range(32)]
    assert len(set(bad)) < 4096 and max(bad) >= 4096
    print('PASS modeled grouped publication service rounds per warp/K32: A 16->8, B 32->8')


if __name__ == '__main__':
    check_publication()
    check_global_and_scales()
    check_bank_model()
    control.check_schedule()
    control.check_tails_and_products()
    control.check_operand_exactness()
    print('CPU model only: CUDA parity, race freedom, graph replay, and timing require GPU validation')
