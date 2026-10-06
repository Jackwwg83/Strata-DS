#!/usr/bin/env python3
"""Compare compiled K1c-09 code with an unchanged control and emit evidence."""
import hashlib
import json
import pathlib
import re
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parents[4]
BASE = 'be8c969a1f1b7bf88d8a64ef1b3e935dcc2f376a'
OUT, CONTROL = map(pathlib.Path, sys.argv[1:3])

def sha(data):
    return hashlib.sha256(data).hexdigest()

def git(*args):
    return subprocess.check_output(['git', '-C', str(SRC), *args])

def entries(path):
    text = path.read_text()
    result = {}
    for match in re.finditer(r'\.entry (\S+)\(', text):
        name = match.group(1)
        a = text.index('{', match.end())
        pos, depth = a + 1, 1
        while depth:
            depth += (text[pos] == '{') - (text[pos] == '}')
            pos += 1
        block = text[match.start():pos]
        normalize = lambda s: re.sub(r'\$L__BB\d+_', '$L__BB_',
            re.sub(r'_GLOBAL__N__[a-z0-9_]+?_fp8_gemv_cu_[0-9a-f]+', '_ANON_', s))
        result[normalize(name)] = normalize(block)
    return result

changed = git('diff', '--name-only', BASE).decode().splitlines()
changed += git('ls-files', '--others', '--exclude-standard').decode().splitlines()
assert all(p == 'src/ds41/kernels/fp8_gemv.cu' or p.startswith('src/ds41/kernels/fp8_gemv/') for p in changed)
protected = ['include/strata/ds41/fp8_gemv.hpp', 'src/ds41/tests/k1c_fp8_gemv_test.cu',
             'src/ds41/kernels/fp8_gemv_parity.cpp', 'src/ds41/ops.cu', 'cmake/ds41.cmake']
protected_hashes = {}
for path in protected:
    data = (SRC / path).read_bytes()
    assert data == git('show', BASE + ':' + path), path
    protected_hashes[path] = sha(data)

source = (SRC / 'src/ds41/kernels/fp8_gemv.cu').read_text()
assert 'M == 1 && SPLIT > 1' in source and 'n * k <= (int64_t(16) << 20)' in source
assert 'if (scale <= 246)' in (SRC / 'src/ds41/kernels/fp8_gemv/small_pair.cuh').read_text()
assert 'return s <= 246' in (SRC / 'src/ds41/kernels/fp8_gemv/check_small_pair.cu').read_text()
for name in ['gemv_split_warps', 'gemv_rows_per_group']:
    assert source.count('detail::' + name) == 1
for n in range(1, 8193):
    split = 4 if n <= 2048 else 2
    rows = 2 if n >= 1024 else 1
    per = 4 // split * rows
    grid = (n + per - 1) // per
    assert grid <= 8192 < 65535 and (grid - 1) * per < n <= grid * per
# Every selected positive dimension and every accessed byte offset is <= 16 MiB.
assert (16 << 20) < (1 << 31)

report = {'base': BASE, 'changed_paths': sorted(set(changed)),
          'protected_sha256': protected_hashes, 'architectures': {},
          'limitations': ['No GPU; numerical/graph/performance acceptance not run',
                          'No nvdisasm in existing toolchain; no SASS instruction validation']}
for arch in [86, 89, 120]:
    folder = OUT / ('sm' + str(arch))
    old = entries(CONTROL / ('sm' + str(arch)) / 'fp8_gemv.ptx')
    new = entries(folder / 'fp8_gemv.ptx')
    same, additions = [], []
    for name, body in old.items():
        assert name in new, ('missing original', arch, name)
        assert body == new[name], ('changed original PTX', arch, name)
        same.append(name)
    for name in new.keys() - old.keys():
        body = new[name]
        assert 'gemv_small_pair' in name
        assert re.search(r'ld\.global(?:\.nc)?\.v4\.u32', body)
        assert re.search(r'ld\.global(?:\.nc)?\.v4\.f32', body)
        assert body.count('bar.sync') == 1
        if arch == 86:
            assert 'prmt.b32' in body and 'cvt.f32.f16' in body
        else:
            assert 'cvt.rn.f16x2.e4m3x2' in body
        assert not re.search(r'\b(?:mma|wgmma|atom)\.', body)
        additions.append(name)
    resources = []
    log = (folder / 'fp8_gemv.compile.log').read_text()
    for name, stack, stores, loads, regs, rest in re.findall(
        r'Function properties for (\S+)\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\s+ptxas info\s*: Used (\d+) registers([^\n]*)', log):
        shared = re.search(r'(\d+) bytes smem', rest)
        assert int(stores) == 0 and int(loads) == 0
        assert not shared or int(shared.group(1)) <= 99 * 1024
        if 'gemv_small_pair' in name:
            resources.append({'name': name, 'registers': int(regs), 'shared_bytes': int(shared.group(1)) if shared else 0,
                              'stack_bytes': int(stack), 'spill_loads': int(loads), 'spill_stores': int(stores)})
    assert len(resources) == 4
    gpu_exit = int((folder / 'gpu-test.exit').read_text())
    assert gpu_exit == 77
    assert 'sm_' + str(arch) in (folder / 'cubins.log').read_text()
    report['architectures'][str(arch)] = {
        'unchanged_original_ptx_entries': len(same), 'new_pair_entries': len(additions),
        'resources': resources, 'gpu_test_exit': gpu_exit,
        'ptx_sha256': sha((folder / 'fp8_gemv.ptx').read_bytes()),
        'object_sha256': sha((folder / 'fp8_gemv.o').read_bytes()),
        'executable_sha256': sha((folder / 'k1c_fp8_gemv_test').read_bytes())}

report['source_sha256'] = {str(p.relative_to(SRC)): sha(p.read_bytes())
    for p in [SRC / 'src/ds41/kernels/fp8_gemv.cu', *sorted((SRC / 'src/ds41/kernels/fp8_gemv').glob('*'))]
    if p.is_file() and p.suffix not in {'.json', '.md'}}
(OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
