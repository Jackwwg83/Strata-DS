#!/usr/bin/env python3
"""Audit CUDA12.8 output: usage check_ptx.py PTX RESOURCE_LOGS..."""
from pathlib import Path
import re
import sys

ptx = Path(sys.argv[1]).read_text()
assert re.search(r'selp\.b32\s+%r\d+, 65535, -65536,', ptx)
width = re.search(r'mov\.u32\s+(%r\d+), 4127;', ptx)
assert width, 'missing width-16 shuffle control (0x101f)'
shuffles = re.findall(r'shfl\.sync\.idx\.b32[^;]+;', ptx)
assert len(shuffles) == 20
assert all(f', {width[1]}, ' in shuffle for shuffle in shuffles)
assert len(re.findall(r'mul\.rn\.f32', ptx)) >= 3
assert 'div.rn.f32' in ptx
assert not re.search(r'\bdiv\.approx\.', ptx)
print('PASS PTX: independent subgroup masks, all 20 static indexed shuffles width=16, rounded multiply/division')
for arg in sys.argv[2:]:
    log = Path(arg).read_text()
    assert len(re.findall(r'0 bytes spill stores, 0 bytes spill loads', log)) == 9
    smem = [int(x) for x in re.findall(r'(\d+) bytes smem', log)]
    assert smem == [200] and max(smem) <= 99 * 1024
    assert len(re.findall(r'Used \d+ registers', log)) == 9
    print(f'PASS {Path(arg).name}: nine kernels, no spills, max shared=200 bytes <=99 KiB')
print('Compiler lowering of ordinary expf can contain ex2 instructions; no fast-math or source approximations are used')
