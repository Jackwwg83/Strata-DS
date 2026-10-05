#!/usr/bin/env python3
"""Inspect external nvcc -ptx and -Xptxas=-v evidence, not a GPU benchmark."""
import json
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
result = {}
for arch in (86,89,120):
    ptx = (root/f'k7-sm{arch}.ptx').read_text()
    kernels = {}
    for section in re.split(r'(?=\n.entry )', ptx):
        if not section.startswith('\n.entry '): continue
        name = section.splitlines()[1]
        if 'hc_packed_partials' not in name: continue
        m = int(re.search(r'ILi(\d)E',name)[1])
        loads = re.findall(r'\b(ld\.global\.[\w.]+)', section)
        assert loads.count('ld.global.nc.v4.f32')==1
        assert loads.count('ld.global.nc.v2.u32')==m
        assert len(loads)==1+m
        assert 'ld.local' not in section and 'st.local' not in section
        kernels[m] = {'weight_v4':1,'activation_v2':m}
    assert sorted(kernels)==list(range(1,9))
    log = (root/f'resources-sm{arch}.log').read_text()
    for stack,store,load in re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log):
        assert int(stack)==int(store)==int(load)==0
    shared = [int(n) for n in re.findall(r'(\d+) bytes smem',log)]
    assert max(shared)==100 and max(shared)<=99*1024
    result[arch] = kernels
    print(f'PASS sm{arch}: every m emits one vector weight load and m packed activation loads; no local-memory ops, stack or spills; maximum shared 100 bytes')
print('PTX evidence only. Analytic dynamic warp-load issues are 192*80*(1+m) scalar versus 48*80*(1+m) packed; SASS issue counts and GPU timing remain unmeasured.')
