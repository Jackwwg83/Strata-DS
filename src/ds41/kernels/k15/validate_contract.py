#!/usr/bin/env python3
"""Supplemental source/host contracts. No CUDA device tests are simulated."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--build-dir',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[4]
base='c19cf821cf1360db4c4b5239497412fe4dfc4319'
runtime=root/'src/ds41/kernels/k15_hc_prefill.cu'
text=runtime.read_text()
for forbidden in ('cudaMalloc','cudaFree','cudaMemcpy','cudaDeviceSynchronize','cudaStreamSynchronize','cudaEventSynchronize','cudaGetDevice','cudaSetDevice','cudaLaunchHostFunc','hc_mixes_pre('):
    assert forbidden not in text, f'Forbidden call in runtime: {forbidden}'
assert text.count('<<<')==4
assert text.count(', 0, stream>>>')==4
fixed=['include/strata/ds41/kernels/k15_hc_prefill.hpp','src/ds41/tests/k15_hc_prefill_test.cu',
       'src/ds41/tests/bench_util.hpp','src/ds41/ops.cu','src/ds41/kernels/k7_hc.cu',
       'ds41/tasks/K15.md','ds41/tasks/README.md','ds41/tasks/PRIORITIES.md']
fixed += [str(p.relative_to(root)) for p in root.rglob('CMakeLists.txt') if '.git' not in p.parts]
for path in fixed:
    expected=subprocess.check_output(['git','show',f'{base}:{path}'],cwd=root)
    assert (root/path).read_bytes()==expected, f'Fixed file changed: {path}'
changed=subprocess.check_output(['git','diff','--name-only',base],cwd=root,text=True).splitlines()
new=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root,text=True).splitlines()
for name in set(changed+new):
    assert name=='src/ds41/kernels/k15_hc_prefill.cu' or name.startswith('src/ds41/kernels/k15/') or name.startswith('ds41/tasks/K15.'), f'Out-of-scope file: {name}'
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
results=[]
for arch in (86,89,120):
    exe=args.build_dir/f'sm{arch}'/'validate_api'
    for case in ('sizes','m0','mneg','mhigh','null','small','align'):
        p=subprocess.run([str(exe),case],capture_output=True,text=True)
        if case=='sizes':
            assert p.returncode==0 and 'workspace sizes pass' in p.stdout,(case,p.returncode,p.stdout,p.stderr)
        else:
            message='invalid token count' if case.startswith('m') else 'workspace is null, too small or not float-aligned'
            assert p.returncode==-6 and message in p.stderr,(case,p.returncode,p.stdout,p.stderr)
        results.append({'arch':arch,'case':case,'exit':p.returncode,'message':(p.stdout+p.stderr).strip()})
print(json.dumps({'scope':'pass','fixed_files':fixed,'host_contracts':results},indent=2))
print('CONTRACT PASS: scope, fixed-file bytes, stream-only launches and host rejection paths')
