#!/usr/bin/env python3
"""CPU/source checks for K10-04; no CUDA numerical/runtime/performance claim.

The manifest describes every permitted scheduling edit, not a broad region
exclusion. Normalization must recover the exact merged adapter source hash.
The native C++ model is extracted from the actual candidate schedule; only
copy/completion instructions are replaced with delayed/eager symbolic copies.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
VENDOR = ROOT / "third_party/exllamav3_gpu"
KERNEL = VENDOR / "quant/exl3_gemv_kernel.cuh"
BASE_KERNEL_SHA256 = '986288fe5e13c3d3162941cda6f6f83f0967678647908364367eb52e92741681'
PIPELINE_SHA256 = 'd3f4ef9b166cd5a5f8b38827753d2effdd09565db2eb2c5bdd05a48ee91cd97a'
ENTRY_SHA256 = '6126c759d893a59c1956fd31c7cedbcfcc77b9250dbedb4a35aa28afb305afb2'
LAUNCHER_SHA256 = '18dbe5b63530ea17f5307df1b4ec969a14da748213a52fc022e2e8f9d434f0e6'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize_cooperative(text):
    for old, new in json.loads((HERE / "cooperative_schedule.json").read_text()):
        assert text.count(new) == 1, "cooperative schedule changed: " + new[:80]
        text = text.replace(new, old)
    assert digest(text.encode()) == "c567255ae83c34a25932d3dc6bc0d566f16d9bce1e88be122b364c2322130e29", "K10-04 parent source changed"
    return text


def normalized_kernel(text):
    text = normalize_cooperative(text)
    for old, new in json.loads((HERE / "async_schedule.json").read_text()):
        assert text.count(new) == 1, "schedule changed: " + new[:80]
        text = text.replace(new, old)
    assert digest(text.encode()) == BASE_KERNEL_SHA256, "arithmetic or baseline source changed"
    return text


def preservation_and_mutations():
    text = KERNEL.read_text()
    normalized_kernel(text)
    for path, expected in [(HERE / "pipeline.cuh", PIPELINE_SHA256),
                           (ROOT / "src/ds41/kernels/k10_exl3_moe.cu", ENTRY_SHA256),
                           (VENDOR / "quant/exl3_gemv.cu", LAUNCHER_SHA256)]:
        assert digest(path.read_bytes()) == expected, path
    mutations = [
        ("x0 *= 0x83DCD12Du;", "x0 *= 0x83DCD12Cu;"),
        ("const int ks0 = warp * chunk;", "const int ks0 = (15 - warp) * chunk;"),
        ("constexpr int FOLD = CFG == 0 ? 4 : 2;", "constexpr int FOLD = CFG == 0 ? 2 : 2;"),
        ("if (first + d < myn)", "if (first + d <= myn)"),
        ("if (lane < LSTRIDE)", "if (lane <= LSTRIDE)"),
        ("cp.async.wait_group 0;", "cp.async.wait_group 1;"),
        ("((ib / PF) + 1) & 1", "(ib / PF) & 1"),
        ("if (i >= myn) break;", "if (i > myn) break;"),
        ("a01, a23, f0, ch[t][0]", "a01, a23, f1, ch[t][0]"),
        ("if ((d + 1) % FOLD == 0 || i + 1 == myn)", "if ((d + 1) % FOLD == 0)"),
        ("sum += sh_red[j][r][c];", "sum += sh_red[WK - 1 - j][r][c];"),
        ("if (!job.B) return;", "if (false) return;"),
    ]
    for old, new in mutations:
        assert text.count(old) == 1, old
        try:
            normalized_kernel(text.replace(old, new))
        except AssertionError:
            pass
        else:
            raise AssertionError("mutation escaped: " + old)
    assert digest((HERE / "pipeline.cuh").read_bytes().replace(b"(silu * u) * weights[slot]", b"silu * (u * weights[slot])")) != PIPELINE_SHA256
    print(f"PASS mutation sensitivity: {len(mutations)} decode, MMA, fold, guard, reduction and async mutations rejected; activation reassociation rejected")
    print("PASS source identity: only explicit raw-copy scheduling/dispatch changes; activation, Hadamards, workspace, host guards and launch count unchanged")


def native_schedule():
    text = normalize_cooperative(KERNEL.read_text())
    init = text[text.index("        auto stage_b ="):text.index("        FragC_h ch[WNT][2]")]
    loop = text[text.index("        for (int ib = 0; ib < myn;"):text.index("            if constexpr (SMEM_STAGE)", text.index("        for (int ib = 0; ib < myn;"))]
    for old, new in [
        ('asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\\n"\n                                             :: "r"(dst), "l"(src) : "memory");', 'copy(dst, src);'),
        ('asm volatile("cp.async.commit_group;\\n" ::: "memory");', 'commit();'),
        ('asm volatile("cp.async.wait_group 0;\\n" ::: "memory");', 'wait();'),
        ('__syncwarp();', '++barriers;'),
    ]:
        count = init.count(old) + loop.count(old)
        assert count == 1, (old, count)
        init, loop = init.replace(old, new), loop.replace(old, new)
    fold = re.search(r"if \(\(d \+ 1\).*?\n", text).group(0).strip()
    pf_expr = re.search(r"constexpr int PF\s*=\s*([^;]+);", text).group(1)
    fold_expr = re.search(r"constexpr int FOLD\s*=\s*([^;]+);", text).group(1)
    code = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>
using std::size_t;
template<bool ASYNC_STAGE> void check(int myn, int lane, int warp, bool eager) {
    constexpr int CFG=0, WK=16, LOADS=2, LSTRIDE=24;
""" + f"    constexpr int PF={pf_expr}, FOLD={fold_expr};\n" + r"""
    const size_t slice_stride=24*17, group_offset=3*48;
    std::vector<uint32_t> global((myn+1)*slice_stride);
    for (size_t p=0; p<global.size(); ++p) global[p]=uint32_t(p+1);
    const uint32_t* bp=global.data()+group_offset+lane;
    uint32_t sh_async[2][WK][PF][LOADS*LSTRIDE];
    std::fill_n(&sh_async[0][0][0][0], 2*WK*PF*LOADS*LSTRIDE, 0xfafafafa);
    std::vector<int> visits(myn*LOADS,0), consumed, folds;
    std::vector<std::pair<unsigned,uint32_t>> pending;
    int barriers=0, commits=0, waits=0;
    auto __cvta_generic_to_shared = [&](const uint32_t* p) {
        return unsigned(p - &sh_async[0][0][0][0]);
    };
    auto address = [&](const uint32_t* src) {
        const ptrdiff_t relative=src-bp;
        const int i=int(relative/slice_stride), l=int((relative%slice_stride)/LSTRIDE);
        assert(relative>=0 && i<myn && l<LOADS && lane<LSTRIDE);
        assert(relative==i*ptrdiff_t(slice_stride)+l*LSTRIDE);
        ++visits[i*LOADS+l];
    };
    auto copy = [&](unsigned dst, const uint32_t* src) {
        address(src);
        assert(dst < 2*WK*PF*LOADS*LSTRIDE);
        assert((dst/(PF*LOADS*LSTRIDE))%WK==unsigned(warp));
        assert(dst%LSTRIDE==unsigned(lane));
        pending.emplace_back(dst,*src);
        if(eager) (&sh_async[0][0][0][0])[dst]=*src;
    };
    auto commit = [&]() { ++commits; };
    auto wait = [&]() {
        ++waits;
        for (auto p:pending) (&sh_async[0][0][0][0])[p.first]=p.second;
        pending.clear();
    };
    auto ld_b = [&](int i,int l)->uint32_t {
        if(lane>=LSTRIDE) return 0;
        const uint32_t* src=bp+i*slice_stride+l*LSTRIDE;
        address(src); return *src;
    };
""" + init + loop + r"""
            for (int l=0;l<LOADS;++l) {
                const uint32_t want=lane<LSTRIDE ? bp[i*slice_stride+l*LSTRIDE] : 0;
                assert(bw[l]==want);
            }
            consumed.push_back(i);
""" + "            " + fold + r""" { folds.push_back(i+1); }
        }
        }
    assert(pending.empty());
    assert(int(consumed.size())==myn);
    for(int i=0;i<myn;++i) assert(consumed[i]==i);
    for(int n:visits) assert(n==(lane<LSTRIDE ? 1:0));
    std::vector<int> expected;
    for(int i=FOLD;i<=myn;i+=FOLD) expected.push_back(i);
    if(myn%FOLD) expected.push_back(myn);
    assert(folds==expected);
    assert(barriers==(ASYNC_STAGE ? (myn+PF-1)/PF : 0));
    assert(waits==barriers);
    assert(commits==(ASYNC_STAGE ? std::max(1,(myn+PF-1)/PF) : 0));
}
int main() {
    // Every lane and warp, legal 9/20-slice chunks, zero/tail chunks and
    // deliberately early and late async completion. Also check the fallback.
    for(int n=0;n<=257;++n) for(int lane=0;lane<32;++lane) {
        const int warp=(n+lane)%16;
        check<true>(n,lane,warp,false);
        check<true>(n,lane,warp,true);
        check<false>(n,lane,warp,false);
    }
    for(int warp=0;warp<16;++warp) for(int n:{9,20}) for(int lane=0;lane<32;++lane) {
        check<true>(n,lane,warp,false); check<true>(n,lane,warp,true);
    }
    std::puts("PASS actual source schedule: 26816 lane/tail/eager/delayed/fallback cases; words once, no invalid copy/read, exact slice/fold order");
}
"""
    with tempfile.TemporaryDirectory(prefix="k10-async-schedule-") as name:
        tmp=Path(name)
        (tmp / "schedule.cpp").write_text(code)
        subprocess.run([os.environ.get("CXX", "clang++"), "-std=c++17", "-O2", "-Wno-unknown-pragmas",
                        str(tmp / "schedule.cpp"), "-o", str(tmp / "schedule")], check=True)
        subprocess.run([str(tmp / "schedule")],check=True)


if __name__ == "__main__":
    preservation_and_mutations()
    native_schedule()
