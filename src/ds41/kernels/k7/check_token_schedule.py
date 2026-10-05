#!/usr/bin/env python3
"""CPU source/ownership checks for K7-09; no CUDA code is executed."""
from pathlib import Path
import subprocess

source = Path(__file__).resolve().parent.parent / 'k7_hc.cu'
s = source.read_text()
control = subprocess.check_output([
    'git', '-C', str(source.parent), 'show',
    '5dc13d8d983c3abad1f9853b2dfa6731adc77dde:src/ds41/kernels/k7_hc.cu'
], text=True)
# Workspace allocation/lifetime and every finish-stage operation are unchanged.
for start, end in [
    ('struct Workspace {', '// Each warp-only CTA'),
    ('// All 32 lanes execute each shuffle.', '}  // namespace\n'),
]:
    assert s[s.index(start):s.index(end)] == control[control.index(start):control.index(end)]
assert 'constexpr int kTokenTile = 4;' in s
assert 'constexpr int TileTokens = Tokens < kTokenTile ? Tokens : kTokenTile;' in s
assert 'const int first_token = Tokens <= kTokenTile ? 0 : blockIdx.z * kTokenTile;' in s
assert 'float dots[TileTokens] = {};' in s and 'float squares[TileTokens] = {};' in s
assert s.count('if (token < Tokens)') == 2
assert 'dots[local_token] = fmaf(value, weight, dots[local_token]);' in s
assert 'squares[local_token] = fmaf(value, value, squares[local_token]);' in s
assert 'const dim3 partial_grid(kDotWarps, kHcMix, (m + kTokenTile - 1) / kTokenTile);' in s
code = '\n'.join(line.split('//')[0] for line in s.splitlines())
assert code.count('cudaMalloc(') == 1
assert not any(x in code for x in ['cudaFree', 'cudaMemcpy', 'cudaStreamSynchronize', 'cudaDeviceSynchronize', 'atomic', '__threadfence'])
assert 'hc_partials<TOKENS><<<partial_grid, kProducerThreads, 0, stream>>>' in s
assert 'hc_finish<<<dim3(kDim / kThreads, m), kThreads, 0, stream>>>' in s

# Each lane's original coordinate sequence is unchanged, including norm lanes.
for row in range(24):
    for warp in range(8):
        for lane in range(32):
            cols = [warp * 32 + lane + step * 256 for step in range(80)]
            assert cols == list(range(warp * 32 + lane, 20480, 256))
            if row < 4:
                norm_cols = [c for step, c in enumerate(cols) if (step & 3) == row]
                assert norm_cols == list(range(row * 256 + warp * 32 + lane, 20480, 1024))

# Exhaust token dispatch, final partial ownership, guarded tails, and collapse.
for m in range(1, 9):
    dots, squares, tokens = set(), set(), []
    groups = (m + 3) // 4
    for group in range(groups):
        for local in range(min(m, 4)):
            token = (0 if m <= 4 else group * 4) + local
            if token >= m:
                continue
            tokens.append(token)
            for row in range(24):
                for warp in range(8):
                    key = (token, row, warp)
                    assert key not in dots
                    dots.add(key)
                    if row < 4:
                        key = (token, row * 8 + warp)
                        assert key not in squares
                        squares.add(key)
    assert tokens == list(range(m))
    assert len(dots) == m * 24 * 8 and len(squares) == m * 32
    assert all(0 <= t < m for t, _, _ in dots)
    assert all(0 <= t < m for t, _ in squares)
    writes = {(token, block * 256 + lane)
              for token in range(m) for block in range(20) for lane in range(256)}
    assert len(writes) == m * 5120
    weight_bytes, x_bytes, scratch_bytes = groups * 24 * 20480 * 4, m * 24 * 20480 * 2, m * 224 * 4
    print(f'm={m} groups={groups} producer_ctas={groups*192} accumulator_pairs={min(m,4)} '
          f'fn_read_bytes={weight_bytes} x_read_bytes={x_bytes} partial_write_bytes={scratch_bytes} partial_read_bytes={scratch_bytes}')
print('PASS source equivalence: workspace, finish helpers, coefficient math, ordered reductions and collapse unchanged from repaired K7-02')
print('PASS coordinate order: 6144 dot chains, 1024 norm chains; unique partial/collapse ownership and guarded tails for every m=1..8')
print('These source-logical bytes exclude cache effects; CUDA correctness, graph execution and timing remain untested')
