#!/usr/bin/env python3
"""Static bounds/order/ownership checks. This does not execute CUDA."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[4]
source = (root / 'src/ds41/kernels/k7_hc.cu').read_text()
control = subprocess.check_output(['git', 'show',
    '5dc13d8d983c3abad1f9853b2dfa6731adc77dde:src/ds41/kernels/k7_hc.cu'],
    cwd=root, text=True)
suffix = '// All 32 lanes execute each shuffle.'
assert source[source.index(suffix):] == control[control.index(suffix):]
start, stop = 'struct Workspace {', '// Each warp-only CTA owns'
assert source[source.index(start):source.index('// One row and original warp')] == \
       control[control.index(start):control.index(stop)]
assert source.count('<<<') == 2
assert source.count('cudaMalloc(') == 1
for forbidden in ('cudaMemcpy', 'cudaStreamSynchronize', 'cudaDeviceSynchronize',
                  'cudaFree', 'cudaMemset', 'cudaMallocAsync', 'cudaMallocFromPoolAsync',
                  '__threadfence', 'atomicAdd', 'cuda::atomic', 'while ('):
    assert forbidden not in source, forbidden
assert 'sizeof(Workspace) == 7168' in source
assert 'if (entry.device == device) return entry.workspace;' in source
print('PASS unchanged repaired-control workspace, finish math, collapse and host launch path')

# Explicitly model the current/future pairs including the separate drain.
current = [0, 1]
loads, consumed = [0, 1], []
for step in range(0, 78, 2):
    future = [step + 2, step + 3]
    loads += future
    assert current == [step, step + 1]
    consumed += current
    current = future
consumed += current
assert loads == consumed == list(range(80))

for m in range(1, 9):
    writers, weights, norm = set(), set(), set()
    for row in range(24):
        for warp in range(8):
            for lane in range(32):
                original = warp * 32 + lane
                columns = [original + step * 256 for step in consumed]
                assert columns == list(range(original, 20480, 256))
                for col in columns:
                    key = row, col
                    assert key not in weights
                    weights.add(key)
                if row < 4:
                    columns = [original + step * 256 for step in consumed if step % 4 == row]
                    assert columns == list(range(row * 256 + original, 20480, 1024))
                    for col in columns:
                        assert col not in norm
                        norm.add(col)
            for token in range(m):
                keys = [(token, row, warp)]
                if row < 4: keys.append((token, 24, row * 8 + warp))
                for key in keys:
                    assert key not in writers
                    writers.add(key)
    assert len(weights) == 24 * 20480 and len(norm) == 20480
    assert len(writers) == m * (24 * 8 + 32)
    outputs = set()
    for token in range(m):
        for tile in range(20):
            for tid in range(256):
                d = tile * 256 + tid
                for j in range(4): assert 0 <= token * 20480 + j * 5120 + d < m * 20480
                key = token, d
                assert key not in outputs
                outputs.add(key)
    assert len(outputs) == m * 5120
print('PASS all m=1..8: original stride-256 FMA chains, stride-1024 norm chains, exclusive outputs')
print('PASS each weight and each RMS input has one owner; no drain overread or scratch clearing')
print('Source/model evidence only; no GPU racecheck or graph replay claim')
