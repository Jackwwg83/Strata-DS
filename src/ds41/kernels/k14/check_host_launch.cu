// Supplementary check: CUDA launch APIs are mocked. This exercises real host planning,
// not CUDA execution, stream semantics, graph capture, or memory safety on a GPU.
#include "strata/ds41/kernels/k14_indexer_prefill.hpp"
#include <array>
#include <cassert>
#include <climits>
#include <cstdio>
#include <vector>
#include <stdexcept>
namespace kk = strata::ds41::kernels;
struct Launch { dim3 grid, threads; cudaStream_t stream; int first, rows; bool score, candidate; const void* scratch; int64_t stride; };
std::vector<Launch> launches;
dim3 cfg_grid, cfg_block;
size_t cfg_shared;
cudaStream_t cfg_stream;
size_t memset_bytes;
cudaStream_t memset_stream;
int memsets, attributes;
const void* configured;
cudaError_t attribute_result = cudaSuccess;
extern "C" cudaError_t __wrap_cudaFuncSetAttribute(const void* fn, cudaFuncAttribute attr, int bytes) {
    assert(attr == cudaFuncAttributeMaxDynamicSharedMemorySize && bytes == 100608);
    configured = fn; ++attributes; return attribute_result;
}
extern "C" unsigned __wrap___cudaPushCallConfiguration(dim3 g, dim3 b, size_t s, cudaStream_t stream) {
    cfg_grid=g; cfg_block=b; cfg_shared=s; cfg_stream=stream; return 0;
}
extern "C" cudaError_t __wrap___cudaPopCallConfiguration(dim3* g, dim3* b, size_t* s, void* stream) {
    *g=cfg_grid; *b=cfg_block; *s=cfg_shared; *static_cast<cudaStream_t*>(stream)=cfg_stream; return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetLastError() { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaMemsetAsync(void*, int v, size_t n, cudaStream_t s) {
    assert(v==255); memset_bytes=n; memset_stream=s; ++memsets; return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* fn, dim3 g, dim3 b, void** args, size_t shared, cudaStream_t stream) {
    assert(shared==(b.x==512 ? 100608 : 0));
    assert(b.x!=512 || (attributes==1 && configured==fn));
    Launch x{}; x.grid=g; x.threads=b; x.stream=stream; x.score=b.x==512;
    if (x.score) {
        x.first=*static_cast<int*>(args[3]); x.rows=*static_cast<int*>(args[4]);
        x.scratch=*static_cast<void**>(args[9]); x.stride=*static_cast<int64_t*>(args[10]);
    } else {
        assert(b.x==256);
        x.first=*static_cast<int*>(args[2]); x.rows=g.x;
        x.scratch=*static_cast<void**>(args[0]); x.stride=*static_cast<int64_t*>(args[1]);
        x.candidate=*static_cast<void**>(args[10])!=nullptr;
    }
    launches.push_back(x); return cudaSuccess;
}
int main() {
    alignas(256) std::array<unsigned char,1024> memory{};
    auto* input=reinterpret_cast<const __nv_bfloat16*>(memory.data());
    auto* out=reinterpret_cast<int32_t*>(memory.data());
    auto stream=reinterpret_cast<cudaStream_t>(uintptr_t(0xabc0));
    struct Case { int m,p,r; };
    const Case shapes[]={{1,0,2},{4,0,INT_MAX},{1,0,1},{3,61,1},{257,0,2},
                         {513,61,7},{4096,4096,1},{16384,INT_MAX,1},{16384,INT_MAX,INT_MAX}};
    int calls=0, nodes=0;
    for (auto c:shapes) for (bool candidate:{false,true}) for (int residue:{0,1,127,255}) {
        launches.clear(); memsets=0; attributes=0;
        int64_t tm=(int64_t(c.p)+c.m)/c.r;
        size_t bytes=kk::indexer_topk_prefill_workspace_bytes(c.m,tm);
        void* ws=tm?memory.data()+residue:nullptr;
        kk::indexer_topk_prefill(tm?input:nullptr,tm?input:nullptr,tm?input:nullptr,
            c.m,c.p,c.r,candidate?nullptr:memory.data(),candidate?memory.data():nullptr,
            tm+3,513,INT_MIN,3,31,out,ws,bytes,stream);
        if (!tm) {
            assert(memsets==1 && attributes==0 && launches.empty());
            assert(memset_bytes==size_t(c.m)*513*sizeof(int32_t) && memset_stream==stream);
        } else {
            assert(memsets==0 && attributes==1);
            size_t idx=0;
            auto aligned=reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(ws)+255)&~uintptr_t(255));
            for (int first=0;first<c.m;first+=256) {
                int rows=std::min(256,c.m-first);
                int64_t end=(int64_t(c.p)+first+rows)/c.r;
                if (end) {
                    auto x=launches.at(idx++);
                    assert(x.score && x.grid.x==unsigned((end+127)/128) && x.grid.y==unsigned((rows+15)/16));
                    assert(x.first==first && x.rows==rows && x.scratch==aligned && x.stride==tm && x.stream==stream);
                }
                for (int part=0;part<1+int(candidate);++part) {
                    auto x=launches.at(idx++);
                    assert(!x.score && x.grid.x==unsigned(rows) && x.grid.y==1 && x.candidate==(part==1));
                    assert(x.first==first && x.rows==rows && x.scratch==aligned && x.stride==tm && x.stream==stream);
                }
            }
            assert(idx==launches.size()); nodes+=int(idx);
        }
        ++calls;
    }
    // Attribute errors must stop before any dependent launch.
    attribute_result=cudaErrorInvalidValue;
    launches.clear(); attributes=0;
    bool failed=false;
    try {
        const size_t bytes=kk::indexer_topk_prefill_workspace_bytes(1,1);
        kk::indexer_topk_prefill(input,input,input,1,0,1,nullptr,nullptr,0,1,0,0,8,
                                out,memory.data(),bytes,stream);
    } catch (const std::runtime_error&) { failed=true; }
    assert(failed && attributes==1 && launches.empty());
    std::printf("PASS: %d real host calls / %d mocked launch nodes; shapes, stream, causal grids, alignment, chunks, zero-context memset\n",calls,nodes);
}
