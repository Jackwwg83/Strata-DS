#!/usr/bin/env python3
"""Extract the actual split/ring/fold schedule; model only FP32 reassociation.

This is NOT a CUDA execution, tensor-core emulator, golden test, or timing test.
Once symbolic fold groups match, each group receives an arbitrary finite FP16
result, standing for the identical sequence of unmodified MMA instructions.
The model then rounds every actual FP32 fold addition and split reduction.
"""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
KERNEL = ROOT / 'third_party/exllamav3_gpu/quant/exl3_gemv_kernel.cuh'

def expr(src, name):
    return re.search(r'constexpr int ' + name + r'\s*=\s*([^;]+);', src).group(1)

def section(src, first, last):
    return src[src.index(first):src.index(last, src.index(first))]

def main():
    src = KERNEL.read_text()
    init = section(src, '        uint32_t pf[PF][LOADS];', '        FragC_h ch[WNT][2]')
    loop = section(src, '        for (int ib = 0; ib < myn;', '            if constexpr (SMEM_STAGE)')
    partition = section(src, '    const int chunk =', '    const uint32_t* B32')
    condition = re.search(r'if \(\(d \+ 1\).*?\n', src).group(0).strip()
    # Assert source operations modeled below have not silently changed.
    assert 'acc0[t][f].x += __low2float(ch[t][f][0]);' in src
    assert 'acc0[t][f].y += __high2float(ch[t][f][0]);' in src
    assert 'sum += sh_red[j][r][c];' in src
    launcher = (ROOT / 'third_party/exllamav3_gpu/quant/exl3_gemv.cu').read_text()
    pipeline = (ROOT / 'src/ds41/kernels/k10/pipeline.cuh').read_text()
    assert 'if (max_n == 2304)' in launcher
    assert 'check_proj(p, H, F);' in pipeline
    assert 'constexpr int H = 5120;' in pipeline and 'constexpr int F = 2304;' in pipeline
    constants = '\n'.join('    constexpr int '+n+'='+expr(src,n)+';' for n in ('WK','WNT','PF','FOLD','THREADS','COLS'))
    source = r'''#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <vector>
using std::min;
using std::max;
#define CEIL_DIVIDE(a,b) (((a)+(b)-1)/(b))
using Fold=std::vector<int>;
using Schedule=std::vector<std::vector<Fold>>;
template<bool KSPLIT8> Schedule schedule(int kslices) {
    constexpr int CFG=0;
''' + constants + r'''
    constexpr int LOADS=WNT;
    Schedule result(WK);
    std::vector<int> visits(kslices,0);
    for (int warp=0;warp<WK;++warp) {
''' + partition + r'''
        std::vector<int> loads(myn*LOADS,0);
        auto ld_b=[&](int i, int l)->uint32_t {
            assert(i>=0 && i<myn && l>=0 && l<LOADS);
            ++loads[i*LOADS+l]; return i*LOADS+l;
        };
        Fold fold;
''' + init + loop + r'''
            for(int l=0;l<LOADS;++l) assert(bw[l] == uint32_t(i*LOADS+l));
            assert(ks0+i>=0 && ks0+i<kslices);
            ++visits[ks0+i]; fold.push_back(ks0+i);
            ''' + condition + r''' {
                result[warp].push_back(fold); fold.clear();
            }
        }
        }
        assert(fold.empty());
        for(int count:loads) assert(count==1);
    }
    for(int count:visits) assert(count==1);
    return result;
}
std::vector<Fold> flatten(const Schedule& s) {
    std::vector<Fold> v; for(const auto& w:s) for(const auto& f:w) v.push_back(f); return v;
}
float fadd(float a,float b) { volatile float v=a+b; return v; }
float evaluate(const Schedule& s,const std::map<Fold,float>& value) {
    float sum=0;
    for(const auto& warp:s) {
        float acc=0;
        for(const auto& fold:warp) acc=fadd(acc,value.at(fold));
        sum=fadd(sum,acc);
    }
    return sum;
}
float half_value(uint16_t u) {
    int exponent=(u>>10)&31, mantissa=u&1023;
    float v=exponent ? std::ldexp(float(1024+mantissa), exponent-25)
                     : std::ldexp(float(mantissa),-24);
    return (u&32768) ? -v : v;
}
void rounding(const Schedule& a,const Schedule& b,int kind,int cases) {
    const auto folds=flatten(a); std::mt19937 rng(39051+kind);
    std::map<Fold,float> values;
    long double square_diff=0,square_ref=0; double max_abs=0,max_rel=0;
    int changed=0,nonfinite=0;
    for(int i=0;i<cases;++i) {
        for(const auto& fold:folds) {
            uint16_t bits;
            if(kind==0) bits=uint16_t((rng()&0x8000)|((10+rng()%11)<<10)|(rng()&1023));
            else bits=uint16_t((rng()&0x8000)|((rng()%31)<<10)|(rng()&1023));
            values[fold]=half_value(bits);
        }
        float x=evaluate(a,values),y=evaluate(b,values);
        if(!std::isfinite(x)||!std::isfinite(y)) { ++nonfinite; continue; }
        double d=double(y)-x; changed+=(x!=y); square_diff+=d*d; square_ref+=double(x)*x;
        max_abs=std::max(max_abs,std::abs(d));
        if(x!=0) max_rel=std::max(max_rel,std::abs(d/x));
    }
    std::printf("FP32 model distribution=%s cases=%d changed=%d rel_l2=%.9g max_abs=%.9g max_scalar_relative=%.9g nonfinite=%d\n",
                kind ? "all_finite_half_exponents":"bounded_half_exponents_10_20",cases,changed,
                std::sqrt(double(square_diff/square_ref)),max_abs,max_rel,nonfinite);
    assert(nonfinite==0);
}
int main() {
    for(int k=0;k<=32768;k+=16) { schedule<false>(k/16); schedule<true>(k/16); }
    std::puts("PASS extracted source ring/partition: K=0..32768 in steps16; every slice/load consumed once, no OOB");
    const auto control=schedule<false>(320),candidate=schedule<true>(320);
    assert(flatten(control)==flatten(candidate));
    assert(flatten(control).size()==80);
    for(const auto& f:flatten(control)) assert(f.size()==4);
    std::puts("PASS K5120: identical80 four-slice FP16 fold groups; only FP32 association changes (16x5 to8x10)");
    assert(flatten(schedule<false>(144))!=flatten(schedule<true>(144)));
    std::puts("PASS rejected K2304 split8: FP16 groups differ; production keeps16-warps original");
    rounding(control,candidate,0,100000);
    rounding(control,candidate,1,100000);
    const auto folds=flatten(control);
    std::map<Fold,float> v; for(const auto& f:folds) v[f]=0;
    // Cancellation across the split boundary: the old partials erase the
    // tiny term, whereas the longer new chain cancels first and retains it.
    v[folds[0]]=v[folds[1]]=65504.f;
    v[folds[5]]=v[folds[6]]=-65504.f;
    v[folds[7]]=half_value(1);
    const float x=evaluate(control,v),y=evaluate(candidate,v);
    assert(x==0.f && y==half_value(1));
    std::printf("Adversarial cancellation example: control=%.9g candidate=%.9g (no universal bitwise/relative-error guarantee)\n",x,y);
    std::puts("MODEL COMPLETE: GPU golden at fixed5e-3 tolerance and graph/replay still required; no timing measured");
}
'''
    with tempfile.TemporaryDirectory(prefix='k10-split-') as tmp:
        cpp=Path(tmp)/'check.cpp'; exe=Path(tmp)/'check'
        cpp.write_text(source)
        subprocess.run([os.environ.get('CXX','g++'),'-std=c++17','-O2','-ffp-contract=off','-fno-fast-math','-Wno-unknown-pragmas',str(cpp),'-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True)

if __name__=='__main__': main()
