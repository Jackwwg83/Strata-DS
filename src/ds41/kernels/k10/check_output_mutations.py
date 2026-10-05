#!/usr/bin/env python3
"""Ensure output equivalence/preservation checks reject concrete regressions.
Only temporary copies are mutated; the checkout is never edited.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
PIPE = 'src/ds41/kernels/k10/pipeline.cuh'
HAD = 'src/ds41/kernels/k10/output_hadamard.cuh'
MUTATIONS = [
    ('reset caller output', PIPE, '    for (int j = 0; j < topk; ++j)', '    acc = float4{0, 0, 0, 0};\n    for (int j = 0; j < topk; ++j)'),
    ('reverse slot order', PIPE, 'for (int j = 0; j < topk; ++j)', 'for (int j = topk - 1; j >= 0; --j)'),
    ('skip last slot', PIPE, 'for (int j = 0; j < topk; ++j)', 'for (int j = 0; j + 1 < topk; ++j)'),
    ('alias adjacent lanes', PIPE, 'off + 4 * lane;', 'off + lane;'),
    ('wrong fourth output', PIPE, 'dst[3] = acc.w;', 'dst[3] = acc.z;'),
    ('read empty slot scratch', PIPE, 'if (id < 0) continue;', 'if (id < 0) { acc.x = down[size_t(slot) * H + off]; continue; }'),
    ('remove add rounding barrier', PIPE, 'acc.x = __fadd_rn(acc.x, v.x);', 'acc.x += v.x;'),
    ('drop alignment fallback', PIPE, 'const bool vector_out = (reinterpret_cast<uintptr_t>(out) & 15u) == 0;', 'const bool vector_out = true;'),
    ('swap vector output components', PIPE, '*reinterpret_cast<float4*>(dst) = acc;', '*reinterpret_cast<float4*>(dst) = float4{acc.y, acc.x, acc.z, acc.w};'),
    ('change butterfly sign', HAD, 'float d0 = v0 - v1;', 'float d0 = v0 + v1;'),
    ('wrong scale chunk', HAD, 'int i = blockIdx.y * 32 + t;', 'int i = t;'),
    ('change GEMV bound', 'third_party/exllamav3_gpu/quant/exl3_gemv_kernel.cuh',
     'CFG == 0 ? 2 : 1)', 'CFG == 0 ? 1 : 1)'),
]

def main():
    with tempfile.TemporaryDirectory(prefix='k10-output-mutations-') as name:
        root = Path(name)
        for directory in ['src/ds41/kernels/k10', 'third_party/exllamav3_gpu']:
            shutil.copytree(ROOT / directory, root / directory)
        shutil.copy2(ROOT / 'src/ds41/kernels/k10_exl3_moe.cu', root / 'src/ds41/kernels/k10_exl3_moe.cu')
        for label, rel, old, new in MUTATIONS:
            path = root / rel
            pristine = path.read_text()
            assert old in pristine, label
            path.write_text(pristine.replace(old, new))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([sys.executable, str(ROOT / 'src/ds41/kernels/k10/check_output_host.py'),
                                     '--root', str(root)], capture_output=True, text=True, env=env)
            path.write_text(pristine)
            if result.returncode == 0:
                raise AssertionError('mutation unexpectedly accepted: ' + label)
            if 'error:' in result.stderr or 'FileNotFoundError' in result.stderr:
                raise AssertionError('mutation failed for an unexpected tool/compiler reason: ' + label + '\n' + result.stderr)
            print('PASS rejected mutation: ' + label, flush=True)
    print(f'PASS {len(MUTATIONS)} mutation-sensitive checks; temporary copies only')

if __name__ == '__main__':
    main()
