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
    loops = body.split('.pragma "nounroll";')[1:]
    assert loops
    for loop in loops:
        prefix = loop[:loop.index('fma.rn.f32')]
        assert prefix.count('ld.global.nc.f32') == 2, (m, 'future weights')
        assert prefix.count('ld.global.nc.u16') == 2 * m, (m, 'future input pair')
    assert body.count('shfl.sync.down.b32') >= m * 10
    print(f'PASS m={m}: two future FP32 weights and {2*m} BF16 loads precede current FMAs; no local memory')
assert seen == set(range(1, 9))
print('PASS all 8 producer specializations retain load-ahead in PTX; no SASS or timing claim')
