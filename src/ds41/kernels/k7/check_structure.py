#!/usr/bin/env python3
"""Static bounds/order/ownership checks. This does not execute CUDA."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[4]
source = (root / 'src/ds41/kernels/k7_hc.cu').read_text()
control = subprocess.check_output(['git', 'show',
    'be8c969a1f1b7bf88d8a64ef1b3e935dcc2f376a:src/ds41/kernels/k7_hc.cu'],
    cwd=root, text=True)
def section(text, first, last):
    return text[text.index(first):text.index(last)]
assert section(source, 'struct Workspace {', '// All 32 lanes') .split('// One warp cooperatively')[0] == section(control, 'struct Workspace {', '// All 32 lanes')
assert section(source, '// All 32 lanes', '}  // namespace\n') == section(control, '// All 32 lanes', '}  // namespace\n')
assert source.count('<<<') == 3  # two mutually exclusive producers, one finish
assert source.count('cudaMalloc(') == 1
for forbidden in ('cudaMemcpy', 'cudaStreamSynchronize', 'cudaDeviceSynchronize',
                  'cudaFree', 'cudaMemset', 'cudaMallocAsync', 'cudaMallocFromPoolAsync',
                  '__threadfence', 'atomicAdd', 'cuda::atomic', 'while ('):
    assert forbidden not in source, forbidden
assert 'sizeof(Workspace) == 7168' in source
assert 'if (entry.device == device) return entry.workspace;' in source
assert 'reinterpret_cast<std::uintptr_t>(fn) % 16 == 0' in source
assert source.count(', 0, stream>>>') == 3
assert 'cp.async.cg.shared.global [%0], [%1], 16;' in source
changed = subprocess.check_output(['git', 'diff', '--name-only', 'be8c969a'], cwd=root, text=True).splitlines()
changed += subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=root, text=True).splitlines()
for path in changed:
    assert path == 'src/ds41/kernels/k7_hc.cu' or path.startswith('src/ds41/kernels/k7/'), path
print('PASS byte-identical control scalar fallback, workspace lifecycle, finalization, Sinkhorn and collapse')
print('PASS two stream-ordered runtime launches; aligned-weight selection; no hot-call host synchronization')

# Cooperative four-float copies cover every original warp column exactly once.
# Check addresses as offsets, including only-float-aligned fallback views.
for base in (0, 4, 8, 12, 16, 20, 32):
    async_path = base % 16 == 0
    assert async_path == (base in (0, 16, 32))
    if not async_path: continue
    weights = set()
    for row in range(24):
        for warp in range(8):
            for tile in range(5):
                shared = set()
                for lane in range(32):
                    for copy in range(4):
                        local = lane * 4 + copy * 128
                        col = warp * 32 + (tile * 16 + local // 32) * 256 + local % 32
                        assert (base + 4 * (row * 20480 + col)) % 16 == 0
                        assert local % 4 == 0 and col + 3 < 20480
                        for v in range(4):
                            assert local + v not in shared
                            shared.add(local + v)
                            assert (row, col + v) not in weights
                            weights.add((row, col + v))
                assert shared == set(range(512))
    assert len(weights) == 24 * 20480
print('PASS cooperative 16-byte alignment, exact single ownership, 4 KiB shared bounds and final-tile bounds')

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
