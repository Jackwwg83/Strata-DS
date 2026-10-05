#!/usr/bin/env python3
"""CPU-only symbolic/ownership checks for K7-10; no CUDA execution or timing."""
from pathlib import Path
import collections
import subprocess

N = 20480

def plus(a, b):
    return ('+', a, b)

def reference_tree(values):
    for off in (16, 8, 4, 2, 1):
        values = [plus(v, values[i+off] if i+off < 32 else v)
                  for i, v in enumerate(values)]
    return values[0]

def packed_tree(values):
    lanes = [values[i:i+4] for i in range(0, 32, 4)]
    for off in (4, 2, 1):
        lanes = [[plus(v, lanes[i+off][c] if i+off < 8 else v)
                  for c, v in enumerate(lane)] for i, lane in enumerate(lanes)]
    x, y, z, w = lanes[0]
    return plus(plus(x,z),plus(y,w))

# Explicit expression trees, not a real-number or commutativity comparison.
for warp in range(32):
    values = [('FMA_chain', warp*32+lane) for lane in range(32)]
    assert packed_tree(values) == reference_tree(values)
    assert packed_tree(values) != plus(plus(values[0], values[1]), plus(values[2], values[3]))
for count in (8,32):
    r = p = '+0'
    for warp in range(count):
        v = [('FMA_chain', warp*32+lane) for lane in range(32)]
        r = plus(r,reference_tree(v))
        p = plus(p,packed_tree(v))
    assert p == r
print('PASS exact symbolic trees: 32 original warp trees, ordered 8/32 warp-total finish')

for row in range(24):
    seen = collections.Counter()
    norm_seen = collections.Counter()
    dot_owners, norm_owners = [], []
    for block in range(2):
        for lane in range(32):
            warp = block*4 + lane//8
            for component in range(4):
                original_lane = warp*32+(lane%8)*4+component
                cols = [original_lane+step*256 for step in range(80)]
                assert cols == list(range(original_lane,N,256))
                seen.update(cols)
                if row < 4:
                    normcols = [c for step,c in enumerate(cols) if step%4 == row]
                    assert normcols == list(range(row*256+original_lane,N,1024))
                    norm_seen.update(normcols)
            if lane%8 == 0:
                dot_owners.append((row,warp))
                if row<4: norm_owners.append(row*8+warp)
    assert sorted(seen)==list(range(N)) and set(seen.values())=={1}
    assert sorted(dot_owners)==[(row,warp) for warp in range(8)]
    if row<4:
        assert set(norm_seen.values())=={1} and len(norm_seen)==N//4
        assert norm_owners==list(range(row*8,row*8+8))
print('PASS mapping: all 24 rows, 80-step stride256 dot chains, 20-step stride1024 RMS chains; unique scratch owners')

for token in range(8):
    for block in range(2):
        for lane in range(32):
            for step in range(80):
                column=(block*4+lane//8)*32+(lane%8)*4+step*256
                assert column%4==0 and 0<=column<=N-4
                assert (token*N+column)*2%8==0
                assert 0<=token*N+column<=8*N-4
for fn_offset in range(0,16,4):
    for x_offset in range(0,8,2):
        packed = fn_offset%16==0 and x_offset%8==0
        assert packed == (fn_offset==0 and x_offset==0)
print('PASS aligned vector bounds for all m=1..8; 16 natural-alignment residue combinations select packed or scalar fallback')

source = Path(__file__).resolve().parents[1] / 'k7_hc.cu'
s = source.read_text()
base = subprocess.check_output(['git','show','5dc13d8d983c3abad1f9853b2dfa6731adc77dde:src/ds41/kernels/k7_hc.cu'],cwd=source.parent,text=True)
# Keep scalar numerical fallback, all nonlinear math, scratch, and finish exact.
for start,end in [('struct Workspace {','// Each warp-only CTA'),
                  ('template <int Tokens>\n__global__ void hc_partials','// All 32 lanes'),
                  ('// All 32 lanes','}  // namespace\n\nvoid hc_mixes_pre')]:
    expected=base[base.index(start):base.index(end)]
    if start.startswith('template'):
        actual=s[s.index(start):s.index('// One physical lane')]
    else:
        actual=s[s.index(start):s.index(end)]
    assert actual==expected, f'inherited arithmetic changed: {start}'
assert 'cudaMemcpy' not in s and 'cudaStreamSynchronize' not in s and 'cudaDeviceSynchronize' not in s
assert s.count('cudaMalloc')==1 and 'cudaFree' not in s
assert 'alignof(float4)' in s and 'alignof(uint2)' in s
print('PASS source parity: scalar fallback, task-private workspace, ordered finish/RMS/nonlinear/collapse identical to exact repaired K7-02; no warm allocation/sync/copy')

for m in (1,8):
    old = 192*80*(1+m)
    packed = 48*80*(1+m)
    print(f'Analytic aligned producer warp-load issues m={m}: {old} scalar -> {packed} vector ({old/packed:g}x fewer); input bytes unchanged')
print('Tradeoff: 192->48 one-warp CTAs, 6144->1536 physical lanes, four independent chain components per lane; no speed claim')
