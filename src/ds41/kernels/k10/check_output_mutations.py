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
    ('reset caller output', PIPE, 'float4 acc{dst[0], dst[1], dst[2], dst[3]};', 'float4 acc{0, 0, 0, 0};'),
    ('reverse slot order', PIPE, 'for (int j = 0; j < topk; ++j)', 'for (int j = topk - 1; j >= 0; --j)'),
    ('skip last slot', PIPE, 'for (int j = 0; j < topk; ++j)', 'for (int j = 0; j + 1 < topk; ++j)'),
    ('alias adjacent lanes', PIPE, 'off + 4 * lane;', 'off + lane;'),
    ('wrong fourth output', PIPE, 'dst[3] = acc.w;', 'dst[3] = acc.z;'),
    ('read empty slot scratch', PIPE, 'if (id < 0) continue;', 'if (id < 0) { acc.x = down[size_t(slot) * H + off]; continue; }'),
    ('remove add rounding barrier', PIPE, 'acc.x = __fadd_rn(acc.x, v.x);', 'acc.x += v.x;'),
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
            print('PASS rejected mutation: ' + label, flush=True)
    print(f'PASS {len(MUTATIONS)} mutation-sensitive checks; temporary copies only')

if __name__ == '__main__':
    main()
