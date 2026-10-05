#!/usr/bin/env python3
"""Static source, paired-token ownership and mask checks. Does not execute CUDA."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[4]
source = (root / 'src/ds41/kernels/k7_hc.cu').read_text()
control = subprocess.check_output(['git', 'show',
    '5dc13d8d983c3abad1f9853b2dfa6731adc77dde:src/ds41/kernels/k7_hc.cu'],
    cwd=root, text=True)
assert source[source.index('#include'):source.index('// Each independent 16-lane')] == \
       control[control.index('#include'):control.index('// All 32 lanes')]
# Host launcher changes only the paired finish grid and its runtime m argument.
host = source[source.index('void hc_mixes_pre('):]
host = host.replace('(m + 1) / 2', 'm').replace('x, m, scale', 'x, scale')
assert host == control[control.index('void hc_mixes_pre('):]
assert source.count('<<<') == 2 and source.count('cudaMalloc(') == 1
for forbidden in ('cudaMemcpy', 'cudaStreamSynchronize', 'cudaDeviceSynchronize',
                  'cudaFree', 'cudaMemset', 'cudaMallocAsync', 'cudaMallocFromPoolAsync',
                  '__threadfence', 'atomicAdd', 'cuda::atomic', '__expf', '__fdividef'):
    assert forbidden not in source, forbidden
assert 'sizeof(Workspace) == 7168' in source
assert 'if (entry.device == device) return entry.workspace;' in source
assert '(threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu' in source
assert source.count('__shfl_sync(mask,') == 3
assert source.count('+ k, 16)') == 2
assert 'j * kHc + (lane & 3), 16)' in source
assert 'iteration < kSinkhornIters - 1' in source
assert 'c = c / row_sum(c, lane, mask) + kHcEps;' in source
assert 'c = c / (row_sum(c, lane, mask) + kHcEps);' in source
assert source.count('c = c / (column_sum(c, lane, mask) + kHcEps);') == 2
print('PASS unchanged repaired producer/workspace and two supplied-stream launches')

for m in range(1, 9):
    producers = {(t, r, w) for t in range(m) for r in range(24) for w in range(8)}
    norms = {(t, w) for t in range(m) for w in range(32)}
    reads, norm_reads, shared, coeff, outputs = set(), set(), set(), set(), set()
    for pair in range((m + 1) // 2):
        for tile in range(20):
            for tid in range(256):
                d = tile * 256 + tid
                for member in range(2):
                    token = pair * 2 + member
                    if token >= m:
                        continue
                    for j in range(4):
                        assert 0 <= token * 20480 + j * 5120 + d < m * 20480
                        assert 0 <= token * 4 + j < m * 4
                    key = token, d
                    assert key not in outputs
                    outputs.add(key)
                if tile != 0:
                    continue
                if tid < 50:
                    member, row = divmod(tid, 25)
                    token = pair * 2 + member
                    if token < m:
                        key = token, row
                        assert key not in shared
                        shared.add(key)
                        if row == 24:
                            for w in range(32):
                                key = token, w
                                assert key in norms and key not in norm_reads
                                norm_reads.add(key)
                        else:
                            for w in range(8):
                                key = token, row, w
                                assert key in producers and key not in reads
                                reads.add(key)
            if tile != 0:
                continue
            # All threads have reached the one barrier before any coefficient.
            for tid in range(32):
                member, lane = divmod(tid, 16)
                token = pair * 2 + member
                if token >= m:
                    continue
                assert all((token, r) in shared for r in range(25))
                mask = 0xffff0000 if tid & 16 else 0x0000ffff
                participants = {i for i in range(32) if mask & (1 << i)}
                assert len(participants) == 16
                assert all(pair * 2 + i // 16 == token for i in participants)
                assert all(pair * 2 + i // 16 < m for i in participants)
                # width=16 resolves the source relative to this subgroup.
                sources = [(lane & 12) + k for k in range(4)]
                sources += [j * 4 + (lane & 3) for j in range(4)]
                for relative in sources:
                    physical = (tid & 16) + relative
                    assert physical in participants
                    assert physical // 16 == member
                keys = [(token, 8 + lane)]
                if lane < 4:
                    keys += [(token, lane), (token, lane + 4)]
                for key in keys:
                    assert key not in coeff
                    coeff.add(key)
    assert reads == producers and norm_reads == norms
    assert len(shared) == m * 25 and len(coeff) == m * 24
    assert len(outputs) == m * 5120
    print(f'PASS m={m}: {20*((m+1)//2)} finish CTAs, unique outputs, all scratch initialized, valid masks')
print('Source/model evidence only; GPU racecheck, graph replay and timing remain untested')
