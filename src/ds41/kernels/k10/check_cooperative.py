#!/usr/bin/env python3
"""K10-09 CPU checks. No GPU numerical, racecheck or timing claim.

Extract and compile the actual cooperative copy lambda. An independent byte
address/ownership oracle checks all lanes, natural source alignments and tails.
The event model separately distinguishes per-thread wait from warp publication
and previous-buffer retirement; source mutations must invalidate the audit.
"""
from pathlib import Path
import os
import subprocess
import tempfile

from check_async import KERNEL, normalized_kernel


def source_and_mutations():
    text = KERNEL.read_text()
    normalized_kernel(text)
    changes = [
        ('& 15u) == 0', '& 7u) == 0'),
        ('& 3u) == 0', '& 1u) == 0'),
        ('if ((lane & 3) == 0)', 'if ((lane & 1) == 0)'),
        ('__shared__ __align__(16) uint32_t sh_async', '__shared__ uint32_t sh_async'),
        ('cp.async.cg.shared.global [%0], [%1], 16;', 'cp.async.cg.shared.global [%0], [%1], 8;'),
        ('cp.async.ca.shared.global [%0], [%1], 4;', 'cp.async.ca.shared.global [%0], [%1], 16;'),
        ('if (lane < LSTRIDE)', 'if (lane <= LSTRIDE)'),
        ('if (first + d < myn)', 'if (first + d <= myn)'),
        ('l * LSTRIDE + lane]', 'l * LSTRIDE + (lane & ~3)]'),
        ('uint32_t(src16[0]) | (uint32_t(src16[1]) << 16)', 'uint32_t(src16[1]) | (uint32_t(src16[0]) << 16)'),
        ('cp.async.wait_group 0;', 'cp.async.wait_group 1;'),
        ('            __syncwarp();\n            if (ib + PF < myn)', '            if (ib + PF < myn)'),
        ('((ib / PF) + 1) & 1', '(ib / PF) & 1'),
        ('sh_async[(ib / PF) & 1]', 'sh_async[((ib / PF) + 1) & 1]'),
        ('if (!job.B) return;', 'if (false) return;'),
    ]
    for old, new in changes:
        assert old in text, old
        try:
            normalized_kernel(text.replace(old, new))
        except AssertionError:
            continue
        raise AssertionError('mutation escaped: ' + old)
    # Temporal order is separate from copy width. The unchanged K10-04 loop
    # waits, publishes/retires, issues the next buffer, then reads this buffer.
    loop = text[text.index('        for (int ib = 0; ib < myn;'):]
    assert loop.index('cp.async.wait_group 0;') < loop.index('__syncwarp();') < loop.index('stage_b(ib + PF,') < loop.index('bw[l] = lane < LSTRIDE ? sh_async')
    print(f'PASS cooperative source: {len(changes)} width/alignment/leader/guard/ownership/wait/publication/reuse/null mutations rejected')


def native_copy_model():
    text = KERNEL.read_text()
    begin = text.index('        auto stage_b =')
    end = text.index('        uint32_t pf[PF][LOADS];', begin)
    stage = text[begin:end]
    for old, new in [
        ('asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n"\n                                                     :: "r"(dst), "l"(src) : "memory");', 'copy(dst, src, 16);'),
        ('asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\\n"\n                                                 :: "r"(dst), "l"(src) : "memory");', 'copy(dst, src, 4);'),
        ('asm volatile("cp.async.commit_group;\\n" ::: "memory");', 'commit();'),
    ]:
        assert stage.count(old) == 1, old
        stage = stage.replace(old, new)
    # Instrument only the scalar fallback destination after its actual u16
    # expression. Keep the read, assembly order, guards and address unchanged.
    expr = 'uint32_t(src16[0]) | (uint32_t(src16[1]) << 16);'
    assert stage.count(expr) == 1
    stage = stage.replace(expr, expr + '\n                                    scalar(dst, src);')
    code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
using std::size_t;
struct Pending { unsigned dst; uint32_t word; };
void check(int myn,int warp,int offset,int ntiles,int group,bool eager) {
    constexpr int WK=16, PF=4, LOADS=2, LSTRIDE=24;
    constexpr bool ASYNC_STAGE=true;
    const size_t slice_stride=size_t(ntiles)*24;
    const size_t group_offset=size_t(group)*48;
    const size_t words=size_t(myn)*slice_stride;
    std::vector<std::max_align_t> storage(((words+slice_stride)*4+64)/sizeof(std::max_align_t)+1);
    uint16_t* B=reinterpret_cast<uint16_t*>(storage.data())+offset/2;
    assert((reinterpret_cast<uintptr_t>(B)&15u)==unsigned(offset));
    auto* bytes=reinterpret_cast<unsigned char*>(B);
    for(size_t i=0;i<words*4;++i) bytes[i]=static_cast<unsigned char>(i*73+19);
    struct Shared {
        std::array<uint32_t,16> before;
        alignas(16) uint32_t data[2][WK][PF][LOADS*LSTRIDE];
        std::array<uint32_t,16> after;
    } shared;
    auto& sh_async=shared.data;
    shared.before.fill(0xfedcba98); shared.after.fill(0x89abcdef);
    std::fill_n(&sh_async[0][0][0][0],2*WK*PF*LOADS*LSTRIDE,0xfafafafa);
    const bool copy16=(reinterpret_cast<uintptr_t>(B)&15u)==0;
    const bool copy4=(reinterpret_cast<uintptr_t>(B)&3u)==0;
    std::array<std::vector<Pending>,32> pending;
    std::vector<int> visits(size_t(myn)*LOADS*LSTRIDE,0);
    std::array<int,32> commits{};
    std::array<int,32> order{};
    std::iota(order.begin(),order.end(),0);
    std::mt19937 rng(unsigned(31*myn+warp+offset));
    int lane=0, current_first=0, current_stage=0;
    const uint32_t* bp=nullptr;
    auto __cvta_generic_to_shared = [&](const uint32_t* p) {
        return unsigned(reinterpret_cast<const unsigned char*>(p)-reinterpret_cast<const unsigned char*>(&sh_async[0][0][0][0]));
    };
    auto validate = [&](unsigned dst,const uint32_t* src,int width) {
        const size_t address=reinterpret_cast<const unsigned char*>(src)-bytes;
        assert(address+width<=words*4);
        assert(address%4==0);
        const size_t source_word=address/4;
        const int i=int(source_word/slice_stride), q=int(source_word%slice_stride)-int(group_offset);
        const int l=q/LSTRIDE, owner=q%LSTRIDE;
        assert(i>=current_first && i<std::min(current_first+PF,myn));
        assert(l>=0 && l<LOADS && owner==lane && owner+width/4<=LSTRIDE);
        const unsigned expected=unsigned((((current_stage*WK+warp)*PF+(i-current_first))*LOADS*LSTRIDE+l*LSTRIDE+lane)*4);
        assert(dst==expected);
        assert(dst+width<=sizeof sh_async);
        for(int t=0;t<width/4;++t) ++visits[(i*LOADS+l)*LSTRIDE+lane+t];
    };
    auto copy = [&](unsigned dst,const uint32_t* src,int width) {
        validate(dst,src,width);
        assert(reinterpret_cast<uintptr_t>(src)%width==0 && dst%width==0);
        assert((width==16 && copy16 && lane%4==0) || (width==4 && !copy16 && copy4));
        for(int t=0;t<width/4;++t) {
            uint32_t word; std::memcpy(&word,reinterpret_cast<const unsigned char*>(src)+t*4,4);
            pending[lane].push_back({dst+unsigned(t*4),word});
            if(eager) std::memcpy(reinterpret_cast<unsigned char*>(sh_async)+dst+t*4,&word,4);
        }
    };
    auto scalar = [&](unsigned dst,const uint32_t* src) {
        assert(!copy4); validate(dst,src,4);
    };
    auto commit=[&]() { ++commits[lane]; };
''' + stage + r'''
    auto issue=[&](int first,int stage) {
        current_first=first; current_stage=stage;
        std::shuffle(order.begin(),order.end(),rng);
        for(int l:order) {
            lane=l;
            bp=reinterpret_cast<const uint32_t*>(B)+group_offset+lane;
            stage_b(first,stage);
        }
    };
    auto publish=[&]() {
        // Per-lane wait operations in arbitrary order precede the full-warp
        // barrier. Nonleader waits alone never complete leader transactions.
        std::shuffle(order.begin(),order.end(),rng);
        for(int l:order) {
            for(auto p:pending[l]) std::memcpy(reinterpret_cast<unsigned char*>(sh_async)+p.dst,&p.word,4);
            pending[l].clear();
        }
    };
    issue(0,0);
    std::vector<int> consumed,folds;
    for(int ib=0;ib<myn;ib+=PF) {
        publish();
        if(ib+PF<myn) issue(ib+PF,((ib/PF)+1)&1);
        for(int d=0;d<PF && ib+d<myn;++d) {
            for(int l=0;l<LOADS;++l) for(int c=0;c<32;++c) {
                uint32_t want=0;
                if(c<LSTRIDE) std::memcpy(&want,bytes+4*((ib+d)*slice_stride+group_offset+l*LSTRIDE+c),4);
                const uint32_t got=c<LSTRIDE ? sh_async[(ib/PF)&1][warp][d][l*LSTRIDE+c] : 0;
                assert(got==want);
            }
            consumed.push_back(ib+d);
            if((d+1)%4==0 || ib+d+1==myn) folds.push_back(ib+d+1);
        }
    }
    for(auto& q:pending) assert(q.empty());
    for(int n:visits) assert(n==1);
    for(int l=0;l<32;++l) assert(commits[l]==std::max(1,(myn+PF-1)/PF));
    assert(int(consumed.size())==myn);
    for(int i=0;i<myn;++i) assert(consumed[i]==i);
    for(int i=0;i<int(folds.size());++i) assert(folds[i]==std::min((i+1)*4,myn));
    // Inactive warps, unconsumed tail slices, and both external guards are not
    // written by active copies, even on eager completion.
    for(int s=0;s<2;++s) for(int w=0;w<WK;++w) if(w!=warp)
      for(int d=0;d<PF;++d) for(int q=0;q<LOADS*LSTRIDE;++q) assert(sh_async[s][w][d][q]==0xfafafafa);
    for(auto v:shared.before) assert(v==0xfedcba98);
    for(auto v:shared.after) assert(v==0x89abcdef);
}
int main() {
    size_t cases=0;
    for(int n=0;n<=257;++n) for(int offset=0;offset<16;offset+=2) for(bool eager:{false,true}) {
        check(n,n%16,offset,17,3,eager); ++cases;
    }
    for(int warp=0;warp<16;++warp) for(int offset=0;offset<16;offset+=2)
      for(auto shape:{std::pair<int,int>{20,144},{9,320}}) for(int group:{0,shape.second/2-1}) for(bool eager:{false,true}) {
        check(shape.first,warp,offset,shape.second,group,eager); ++cases;
    }
    std::printf("PASS actual cooperative source: %zu whole-warp cases; every uint16 alignment, every lane/warp, 0..257 tails, eager/delayed completion, exact words/order and canaries\n",cases);
}
'''
    with tempfile.TemporaryDirectory(prefix='k10-cooperative-') as name:
        tmp=Path(name)
        (tmp/'copy.cpp').write_text(code)
        subprocess.run([os.environ.get('CXX','clang++'),'-std=c++17','-O2','-Wall','-Wextra','-Wno-unknown-pragmas',str(tmp/'copy.cpp'),'-o',str(tmp/'copy')],check=True)
        subprocess.run([str(tmp/'copy')],check=True)


def event_dependencies():
    # A leader's wait makes its own four words complete. Nonleaders cannot
    # read those words safely until the leader and all consumers rendezvous.
    # Enumerate possible orderings with one leader and one peer: a missing
    # publication barrier permits peer-read before leader-wait; a missing
    # retirement barrier permits eager overwrite before the peer's old read.
    import itertools
    publish_bad=retire_bad=legal=0
    events=('issue','leader_wait','peer_wait','old_read','next_copy','read')
    for order in itertools.permutations(events):
        p={x:order.index(x) for x in events}
        if not(p['issue']<p['leader_wait'] and p['peer_wait']<p['read'] and p['leader_wait']<p['next_copy']):
            continue
        safe_publish=p['leader_wait']<p['read']
        safe_retire=p['old_read']<p['next_copy']
        publish_bad+=not safe_publish
        retire_bad+=not safe_retire
        if safe_publish and safe_retire:
            legal+=1
    assert publish_bad and retire_bad and legal
    print(f'PASS independent async event model: {publish_bad} missing-publication and {retire_bad} missing-retirement unsafe schedules distinguished from {legal} valid schedules')


if __name__=='__main__':
    source_and_mutations()
    native_copy_model()
    event_dependencies()
