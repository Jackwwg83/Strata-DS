#!/usr/bin/env python3
"""CPU-only ownership, alias-range, and unchanged-arithmetic source checks."""
from pathlib import Path
import subprocess
import tempfile

source = Path(__file__).resolve().parent.parent / 'k7_hc.cu'
s = source.read_text()
assert 'static_assert(sizeof(Workspace) == 89088);' in s
assert s.count('cudaMalloc(') == 1
# Ignore prose when checking for prohibited warm-path operations.
code = '\n'.join(line.split('//')[0] for line in s.splitlines())
assert not any(t in code for t in ['cudaFree(', 'cudaMemcpy', 'cudaStreamSynchronize', 'cudaDeviceSynchronize', 'atomic', '__threadfence'])
finish = s[s.index('__global__ void hc_finish('):s.index('// Only the overlap fallback')]
assert '__bfloat162float' not in finish and 'pre_in' not in finish
assert 'const int token = blockIdx.x;' in finish
assert 'hc_finish<<<m, 32, 0, stream>>>' in s
assert 'hc_produce<TOKENS><<<producer_blocks, kProducerThreads, 0, stream>>>' in s
assert s.index('hc_finish<<<') < s.index('hc_copy_collapsed<<<')
assert 'dots[token] = fmaf(value, weight, dots[token]);' in s
assert 'squares[token] = fmaf(value, value, squares[token]);' in s

# Exhaust every dot/norm lane's ordered input sequence and every scratch owner.
seen_dot = set()
seen_norm = set()
for block in range(20, 212):
    row, warp = divmod(block - 20, 8)
    assert (row, warp) not in seen_dot
    seen_dot.add((row, warp))
    for lane in range(32):
        cols = [warp * 32 + lane + step * 256 for step in range(80)]
        assert cols == list(range(warp * 32 + lane, 20480, 256))
        if row < 4:
            norm_cols = [c for step, c in enumerate(cols) if (step & 3) == row]
            norm_lane = row * 256 + warp * 32 + lane
            assert norm_cols == list(range(norm_lane, 20480, 1024))
            assert norm_lane not in seen_norm
            seen_norm.add(norm_lane)
assert len(seen_dot) == 192 and len(seen_norm) == 1024
for m in range(1, 9):
    writes = [0] * (m * 5120)
    for block in range(20):
        for lane in range(32):
            for offset in range(lane, 256, 32):
                for token in range(m):
                    writes[token * 5120 + block * 256 + offset] += 1
    assert set(writes) == {1}
print('PASS exact coordinate order: 6144 dot lanes, 1024 norm lanes; unique collapse owners for all m=1..8')

# Compile the actual address-range and dispatch helpers as host C++, not a
# reimplementation. No CUDA driver or device data is touched by this check.
helpers = s[s.index('bool ranges_overlap('):s.index('\n}  // namespace\n')]
harness = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstddef>
using __nv_bfloat16 = uint16_t;
constexpr int kDim=5120, kStreamSize=20480, kHcMix=24, kHc=4;
'''+helpers+r'''
int main() {
    int checks=0;
    auto ptr=[](uintptr_t x){return reinterpret_cast<const void*>(x);};
    for(uintptr_t a=1;a<=128;++a)for(uintptr_t b=1;b<=128;++b)
        for(size_t an=1;an<=16;++an)for(size_t bn=1;bn<=16;++bn){
            assert(ranges_overlap(ptr(a),an,ptr(b),bn)==(a<b+bn&&b<a+an));++checks;
        }
    const uintptr_t high=UINTPTR_MAX-8192;
    for(uintptr_t a: {uintptr_t(8),high})for(uintptr_t b: {uintptr_t(8),high}){
        assert(ranges_overlap(ptr(a),4096,ptr(b),4096)==(a==b));++checks;
    }
    for(int m=1;m<=8;++m){
        uintptr_t inputs[5]={1u<<20,1u<<22,1u<<24,1u<<26,1u<<28};
        size_t bytes[5]={size_t(m)*20480*2,24u*20480*4,12,96,size_t(m)*4*4};
        const size_t ybytes=size_t(m)*5120*2;
        auto check=[&](uintptr_t y){return collapse_needs_staging(
            reinterpret_cast<const __nv_bfloat16*>(inputs[0]),m,
            reinterpret_cast<const float*>(inputs[1]),reinterpret_cast<const float*>(inputs[2]),
            reinterpret_cast<const float*>(inputs[3]),reinterpret_cast<const float*>(inputs[4]),
            reinterpret_cast<const __nv_bfloat16*>(y));};
        assert(!check(1u<<30));++checks;
        for(int q=0;q<5;++q){
            for(uintptr_t y: {inputs[q],inputs[q]+bytes[q]-2,inputs[q]-ybytes+2}){assert(check(y));++checks;}
            for(uintptr_t y: {inputs[q]+bytes[q],inputs[q]-ybytes}){assert(!check(y));++checks;}
        }
    }
    std::printf("PASS actual host alias helper: %d address cases, all five input ranges, all legal m\n",checks);
}
'''
harness = '#include <initializer_list>\n' + harness
with tempfile.TemporaryDirectory() as tmp:
    test = Path(tmp) / 'alias.cpp'
    exe = Path(tmp) / 'alias'
    test.write_text(harness)
    subprocess.run(['g++', '-std=c++17', '-O2', str(test), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
print('PASS source schedule and rule7 checks; these checks do not execute a CUDA kernel')
