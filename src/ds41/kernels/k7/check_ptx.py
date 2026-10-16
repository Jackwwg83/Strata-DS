#!/usr/bin/env python3
"""Inspect nvcc -O3 -arch=sm_89 -ptx output; this is not SASS/runtime proof."""
from pathlib import Path
import re
import sys

ptx = Path(sys.argv[1]).read_text()
seen = set()
for body in ptx.split('.entry ')[1:]:
    match = re.match(r'\w*hc_partialsILi([1-8])E', body)
    if not match: continue
    m = int(match[1])
    seen.add(m)
    assert 'ld.local' not in body and 'st.local' not in body and 'bar.sync' not in body
    # each specialization loads its 80 weights and 80*m inputs exactly once (m > 4: the two-step loop, in its body)
    # CUDA 12 prints the loads as .f32/.u16, CUDA 13 as .b32/.b16
    w = body.count('ld.global.nc.f32') + body.count('ld.global.nc.b32')
    x = body.count('ld.global.nc.u16') + body.count('ld.global.nc.b16')
    assert (w, x) == ((80, 80 * m) if m <= 4 else (4, 4 * m)), (m, 'loads', w, x)
    assert body.count('shfl.sync.down.b32') >= m * 10
    print(f'PASS m={m}: {w} weight and {x} input load instructions; no local memory')
assert seen == set(range(1, 9))
print('PASS all 8 producer specializations: every load once, no local memory; no SASS or timing claim')
