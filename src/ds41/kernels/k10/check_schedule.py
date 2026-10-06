#!/usr/bin/env python3
"""Compile the actual source prefetch schedule as a symbolic host C++ model.

No CUDA execution or approximation of tensor-core arithmetic. Exact slice order
and four-slice FP16 fold boundaries are checked before arithmetic preservation is
separately audited by check_host.py against the hash-verified upstream source.
"""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
KERNEL = ROOT / 'third_party/exllamav3_gpu/quant/exl3_gemv_kernel.cuh'


def source_model(text):
    pf = re.search(r'constexpr int PF\s*=\s*([^;]+);', text).group(1)
    fold = re.search(r'constexpr int FOLD\s*=\s*([^;]+);', text).group(1)
    init = text[text.index('        uint32_t pf[PF][LOADS];'):text.index('        FragC_h ch[WNT][2]')]
    start = text.index('        for (int ib = 0; ib < myn;')
    loop = text[start:text.index('            if constexpr (SMEM_STAGE || GENERIC)', start)]
    condition = re.search(r'if \(\(d \+ 1\).*?\n', text).group(0).strip()
    return '''#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
''' + f'''template<int CFG, int LOADS> void check(int myn) {{
    constexpr int PF={pf};
    constexpr int FOLD={fold};
    constexpr int expected_fold=CFG == 0 ? 4 : 2;
    std::vector<int> loads(myn * LOADS, 0), consumed, folds;
    auto ld_b = [&](int i, int l)->uint32_t {{
        assert(i >= 0 && i < myn && l >= 0 && l < LOADS);
        ++loads[i * LOADS + l];
        return uint32_t(i * LOADS + l);
    }};
''' + init + loop + '''
            for (int l=0; l<LOADS; ++l) assert(bw[l] == uint32_t(i * LOADS + l));
            consumed.push_back(i);
''' + condition + ''' { folds.push_back(i+1); }
        }
        }
    assert(int(consumed.size()) == myn);
    for (int i=0; i<myn; ++i) assert(consumed[i] == i);
    for (int n : loads) assert(n == 1);
    std::vector<int> expected;
    for (int i=expected_fold; i<=myn; i+=expected_fold) expected.push_back(i);
    if (myn % expected_fold) expected.push_back(myn);
    assert(folds == expected);
}
int main() {
    for (int n=0; n<=4096; ++n) {
        check<0,1>(n); check<0,2>(n); check<0,4>(n);
        check<1,1>(n); check<1,2>(n); check<1,4>(n);
    }
    for (int k=0; k<=32768; k+=16) {
        const int slices=k/16, chunk=(slices+15)/16;
        std::vector<int> visits(slices,0);
        for (int warp=0; warp<16; ++warp) {
            const int start=warp*chunk;
            const int count=std::max(0, std::min(chunk,slices-start));
            check<0,2>(count);
            for(int i=0;i<count;++i) ++visits[start+i];
        }
        for (int n:visits) assert(n==1);
    }
    std::puts("PASS actual source: 24582 narrow/wide/width tail cases; once-only slices and upstream fold boundaries; all warp partitions for K=0..32768 step16");
}
'''


def run(text, expected_pass=True):
    with tempfile.TemporaryDirectory(prefix='k10-schedule-') as directory:
        tmp = Path(directory)
        (tmp/'model.cpp').write_text(source_model(text))
        subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++17', '-O2',
                        '-Wno-unknown-pragmas', str(tmp/'model.cpp'), '-o', str(tmp/'model')], check=True)
        result = subprocess.run([str(tmp/'model')], capture_output=True, text=True, cwd=tmp)
        assert (result.returncode == 0) == expected_pass, result.stdout + result.stderr
        if expected_pass:
            print(result.stdout.strip())
        else:
            assert 'Assertion' in result.stderr, result.stderr


def main():
    text=KERNEL.read_text()
    run(text)
    mutations = {
        'wrong ring slot': ('bw[l] = pf[d % PF][l];', 'bw[l] = pf[(d + 1) % PF][l];'),
        'wrong refill slot': ('pf[d % PF][l] = ld_b(i + PF, l);', 'pf[(d + 1) % PF][l] = ld_b(i + PF, l);'),
        'wrong refill guard': ('if (i + PF < myn)', 'if (i + PF + 1 < myn)'),
        'wrong refill slice': ('= ld_b(i + PF, l);', '= ld_b(i + PF + 1, l);'),
        'shortened FP16 fold': ('if ((d + 1) % FOLD == 0 || i + 1 == myn)', 'if ((d + 1) % PF == 0 || i + 1 == myn)'),
        'missing tail fold': ('if ((d + 1) % FOLD == 0 || i + 1 == myn)', 'if ((d + 1) % FOLD == 0)'),
    }
    for name,(old,new) in mutations.items():
        assert text.count(old)==1,(name,text.count(old))
        run(text.replace(old,new),False)
        print('PASS schedule model rejects '+name)


if __name__ == '__main__':
    main()
