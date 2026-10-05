#!/usr/bin/env python3
"""CPU-only selection model; this does not claim CUDA runtime validation."""
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    biased: float
    unbiased: float
    id: int


EMPTY = Candidate(-math.inf, 0.0, 2**31 - 1)


def precedes(a, b):
    return a.biased > b.biased or (a.biased == b.biased and a.id < b.id)


def warp_best(lanes):
    lanes = list(lanes)
    assert len(lanes) == 32
    for offset in (16, 8, 4, 2, 1):
        old = lanes[:]
        for lane in range(32):
            other = old[lane + offset] if lane + offset < 32 else old[lane]
            if precedes(other, old[lane]):
                lanes[lane] = other
    return lanes[0]


def producer(group):
    lanes = list(group) + [EMPTY] * 24
    result = []
    for _ in range(6):
        winner = warp_best(lanes)
        result.append(winner)
        lanes = [EMPTY if item.id == winner.id else item for item in lanes]
    return result


def merge(groups):
    candidates = [item for group in groups for item in producer(group)]
    assert len(candidates) == 288
    a = candidates[:256]
    b = candidates[256:] + [EMPTY] * 224
    result = []
    for _ in range(6):
        local = [right if precedes(right, left) else left for left, right in zip(a, b)]
        warps = [warp_best(local[start:start + 32]) for start in range(0, 256, 32)]
        winner = warp_best(warps + [EMPTY] * 24)
        result.append(winner)
        a = [EMPTY if item.id == winner.id else item for item in a]
        b = [EMPTY if item.id == winner.id else item for item in b]
    return result


def check(scores, raw=None):
    if raw is None:
        raw = [float(i % 31) / 8 for i in range(384)]
    row = [Candidate(scores[i], raw[i], i) for i in range(384)]
    got = merge([row[start:start + 8] for start in range(0, 384, 8)])
    want = sorted(row, key=lambda item: (-item.biased, item.id))[:6]
    assert got == want, (got, want)
    # Normalization uses the original unbiased values, in final routing order.
    total = 0.0
    expected_total = 0.0
    for actual, expected in zip(got, want):
        total += actual.unbiased
        expected_total += expected.unbiased
    assert total == expected_total
    weights = [item.unbiased / (total + 1e-20) * 1.5 for item in got]
    expected = [item.unbiased / (expected_total + 1e-20) * 1.5 for item in want]
    assert weights == expected


def main():
    random_source = random.Random(15005)
    rows = 0
    # Degenerate ties, sentinels, low/high IDs, and concentration in each group.
    for value in (0.0, -0.0, 1.0, -math.inf, math.inf):
        check([value] * 384)
        rows += 1
    for group in range(48):
        scores = [-100.0] * 384
        scores[group * 8:group * 8 + 8] = [10.0] * 8
        check(scores)
        rows += 1
    for direction in (1.0, -1.0):
        check([direction * i for i in range(384)])
        rows += 1
    check([0.0] * 384, [0.0] * 384)
    rows += 1
    # Nonlinear values around zero and the reference's softplus threshold.
    logits = [-1000.0, -100.0, -20.0, -0.0, 0.0, 1.0,
              math.nextafter(20.0, -math.inf), 20.0,
              math.nextafter(20.0, math.inf), 1000.0]
    raw = [math.sqrt(z if z > 20.0 else math.log1p(math.exp(z)))
           for z in (logits[i % len(logits)] for i in range(384))]
    check([s + ((i * 19) % 17 - 8) for i, s in enumerate(raw)], raw)
    rows += 1
    # All legal m dispatches, varied group/tie patterns, and adjacent doubles.
    for m in range(1, 9):
        for _ in range(8):
            for token in range(m):
                modes = token % 3
                if modes == 0:
                    scores = [float(random_source.randrange(-8, 9)) for _ in range(384)]
                elif modes == 1:
                    scores = [random_source.uniform(-10, 10) for _ in range(384)]
                else:
                    scores = [1.0 + random_source.randrange(32) * 2**-52 for _ in range(384)]
                check(scores)
                rows += 1
    print(f'PASS: {rows} CPU selection-model rows; all m=1..8; ties, bounds, and normalization')
    print('CUDA execution, graph replay, hardware numerics, and performance are NOT tested.')


if __name__ == '__main__':
    main()
