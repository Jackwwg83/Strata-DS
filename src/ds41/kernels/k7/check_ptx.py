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

async_seen = set()
for body in ptx.split('.entry ')[1:]:
    match = re.match(r'\w*hc_async_partialsILi([1-8])E', body)
    if not match: continue
    m = int(match[1])
    async_seen.add(m)
    for instruction in ('cp.async.cg.shared.global', 'cp.async.commit_group',
                        'cp.async.wait_group 1', 'cp.async.wait_group 0', 'bar.warp.sync',
                        'ld.shared.f32', 'fma.rn.f32'):
        assert instruction in body, (m, instruction)
    for forbidden in ('ld.local', 'st.local', 'bar.sync', 'ld.global.nc.f32', 'ld.global.f32'):
        assert forbidden not in body, (m, forbidden)
    assert body.count('cp.async.cg.shared.global') == 12, (m, 'two prime tiles plus one refill body')
    assert body.count('bar.warp.sync') == 2, (m, 'publication and reuse')
    print(f'PASS async m={m}: 16-byte cp.async, two waits, two warp barriers, shared FP32 loads, no scalar global fn loads/local memory')
assert async_seen == set(range(1, 9))

if len(sys.argv) > 2:
    sass = Path(sys.argv[2]).read_text()
    sass_seen = set()
    for body in sass.split('Function : ')[1:]:
        match = re.match(r'\w*hc_async_partialsILi([1-8])E', body)
        if not match: continue
        m = int(match[1])
        sass_seen.add(m)
        for instruction in ('LDGSTS', 'DEPBAR', 'WARPSYNC'):
            assert instruction in body, (m, instruction)
        print(f'PASS SASS async m={m}: hardware global-to-shared copies, dependency waits and warp synchronization')
    if sass_seen:
        assert sass_seen == set(range(1, 9))
    else:
        print('SKIP SASS instruction check: cuobjdump emitted native cubin metadata only; no disassembler instructions available')
print('Instruction evidence only: GPU parity, sanitizer, capture/replay and speed remain untested')
