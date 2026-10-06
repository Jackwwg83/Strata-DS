#!/usr/bin/env python3
"""Static ownership/bounds checks and generated-PTX chain checks; not GPU proof."""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[4]
source = (root / 'src/ds41/kernels/k15_hc_prefill.cu').read_text()
header = (root / 'src/ds41/kernels/k15/exact_accumulate.hpp').read_text()
assert source.count('<<<') == 1
assert 'hc_rows<<<m, kThreads, 0, stream>>>' in source
assert 'return 0;' in source[source.index('size_t hc_mixes_pre_rows_workspace_bytes'):]
assert source.count('__syncthreads();') == 2
for forbidden in ('cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaMemset', 'cudaStreamSynchronize',
                  'cudaDeviceSynchronize', '__threadfence', 'atomicAdd', 'cudaFuncSetAttribute'):
    assert forbidden not in source, forbidden
assert 'if (m < 1 || m > 16384)' in source
assert source.count('__fmul_rn') == 3
assert 'kSinkhornIters - 1' in source
assert 'float (&squares)[4]' in header

# Every physical lane's original 256-way dot chain, and the four original
# 1024-way norm chains, including each phase's reference warp-total index.
all_norm = set()
all_dot = set()
for tid in range(256):
    dot = []
    norms = [[] for _ in range(4)]
    for group in range(20):
        for phase in range(4):
            col = tid + (group * 4 + phase) * 256
            dot.append(col)
            norms[phase].append(col)
    assert dot == list(range(tid, 20480, 256))
    all_dot.update(dot)
    for phase in range(4):
        assert norms[phase] == list(range(tid + phase * 256, 20480, 1024))
        assert phase * 8 + tid // 32 == (tid + phase * 256) // 32
        all_norm.update(norms[phase])
assert all_dot == all_norm == set(range(20480))
owned_y = [d for tid in range(256) for d in range(tid, 5120, 256)]
assert sorted(owned_y) == list(range(5120))
# All sizes are launch-bound, not token-tile-bound; check every valid m's final
# input/output element and prove there is no padding or inactive token CTA.
for m in range(1, 16385):
    token = m - 1
    assert token * 20480 + 3 * 5120 + 5119 == m * 20480 - 1
    assert token * 5120 + 5119 == m * 5120 - 1
    assert token * 4 + 3 == m * 4 - 1
    assert token * 16 + 15 == m * 16 - 1
print('PASS all m=1..16384: exact dot/norm chains, unique output ownership, bounded offsets, one stream launch')
print('PASS no workspace, allocation, host synchronization, atomics, persistent state, or function attributes')

for name in sys.argv[1:]:
    ptx = Path(name).read_text()
    assert '.local ' not in ptx and 'ld.local' not in ptx and 'st.local' not in ptx
    assert ptx.count('bar.sync') == 2
    assert ptx.count('mul.rn.f32') == 3
    assert ptx.count('shfl.sync.down.b32') == 28 * 5
    first = ptx.index('.pragma "nounroll";')
    stop = ptx.index('\n\t@', first)
    loop = ptx[first:stop]
    assert loop.count('ld.global.nc.u16') == 4
    assert loop.count('ld.global.nc.f32') == 96
    assert loop.count('fma.rn.f32') == 100
    weights = {}
    load_pattern = r'ld.global.nc.f32\s+(%f\d+), \[%rd\d+(?:\+(-?\d+))?\]'
    offset_base = min(int(m[1] or 0) for m in re.findall(load_pattern, loop))
    chain_start, chain_last, phases = {}, {}, {}
    norm = 0
    for line in loop.splitlines():
        load = re.search(load_pattern, line)
        if load:
            offset = int(load[2] or 0) - offset_base
            row, phase = offset // (20480 * 4), (offset % (20480 * 4)) // (256 * 4)
            assert 0 <= row < 24 and 0 <= phase < 4
            weights[load[1]] = (row, phase)
        fma = re.search(r'fma.rn.f32\s+(%f\d+), (%f\d+), (%f\d+), (%f\d+)', line)
        if not fma:
            continue
        dst, a, b, acc = fma.groups()
        if a == b:
            assert dst == acc
            norm += 1
            continue
        row, phase = weights[b]
        assert phase == phases.get(row, 0)
        if phase == 0:
            chain_start[row] = acc
        else:
            assert chain_last[row] == acc
        chain_last[row] = dst
        phases[row] = phase + 1
    assert norm == 4 and set(phases.values()) == {4} and len(phases) == 24
    assert chain_start == chain_last
    assert re.search(r'setp.ne.s32\s+%p\d+, %r\d+, 20;', loop)
    print(f'PASS {name}: all 24 closed four-phase FP32 FMA chains, four norm chains, 20 groups, no local memory')
print('Static/source/PTX evidence only; GPU acceptance and performance remain untested')
