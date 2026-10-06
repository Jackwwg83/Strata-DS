// src/ds41/doorbell.cu - see include/strata/ds41/doorbell.hpp.
#include "strata/ds41/doorbell.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::ds41 {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 doorbell: ") + what + ": " + cudaGetErrorString(e));
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

constexpr size_t kAlign = 256;   // every buffer, and the two words on separate lines
size_t up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

__device__ void rebase(kernels::Exl3Proj& p, const uint8_t* src, uint8_t* dst) {
    p.trellis = reinterpret_cast<const uint16_t*>(dst + (reinterpret_cast<const uint8_t*>(p.trellis) - src));
    p.suh = reinterpret_cast<const __half*>(dst + (reinterpret_cast<const uint8_t*>(p.suh) - src));
    p.svh = reinterpret_cast<const __half*>(dst + (reinterpret_cast<const uint8_t*>(p.svh) - src));
}

__global__ void publish_k(const uint4* __restrict__ x, int n_x16, const int32_t* __restrict__ ids,
                          const float* __restrict__ w, int m, int topk, const int32_t* __restrict__ res,
                          int32_t* __restrict__ gpu_sel, uint4* mx, int32_t* mids, float* mw, volatile uint32_t* seq,
                          uint32_t round, const kernels::Exl3Expert* vram, const kernels::Exl3Expert* ram,
                          const int* quota, kernels::Exl3Expert* call, ExpertDoorbell::Counts* counts,
                          const ExpertBlob* blobs, uint8_t* staging, size_t stride,
                          ExpertCopy* jobs, int* copy_count) {
    for (int i = threadIdx.x; i < n_x16; i += blockDim.x) mx[i] = x[i];
    // At most 48 route uses in decode/verify. One lane makes the prefix rule explicit.
    if (threadIdx.x == 0) {
        ExpertDoorbell::Counts c{};
        const bool descriptors = vram != nullptr || ram != nullptr;
        const int q = quota ? max(0, min(*quota, topk)) : 0;
        for (int t = 0; t < m; ++t) {
            int used = 0;
            for (int j = 0; j < topk; ++j) {
                const int i = t * topk + j;
                const int32_t id = ids[i];
                const int32_t slot = (res && id >= 0) ? res[id] : -1;
                const bool hit = slot >= 0;
                const bool zc = !hit && id >= 0 && ram && used < q && ram[id].w1.trellis;
                mids[i] = (id < 0 || hit || zc) ? -1 : id;
                mw[i] = w[i];
                if (gpu_sel) gpu_sel[i] = descriptors ? ((hit || zc) ? i : -1) : slot;
                if (descriptors) {
                    call[i] = hit ? vram[slot] : (zc ? ram[id] : kernels::Exl3Expert{});
                    if (zc && staging) {
                        const ExpertBlob blob = blobs[id];
                        const auto* src = reinterpret_cast<const uint8_t*>(call[i].w1.trellis) - blob.first_trellis;
                        auto* dst = staging + size_t(i) * stride;
                        jobs[c.zero_copy] = {src, dst, blob.bytes};
                        rebase(call[i].w1, src, dst);
                        rebase(call[i].w3, src, dst);
                        rebase(call[i].w2, src, dst);
                    }
                }
                if (hit) ++c.vram;
                else if (zc) { ++used; ++c.zero_copy; }
                else if (id >= 0) ++c.cpu;
            }
        }
        *counts = c;
        if (copy_count) *copy_count = c.zero_copy;
    }
    __threadfence_system();   // Every thread publishes its writes before seq.
    __syncthreads();
    if (threadIdx.x == 0) {
        *seq = round;
        __threadfence_system();
    }
}

__global__ void wait_add_k(const volatile uint32_t* done, uint32_t round, const float4* y, float4* out, int n4) {
    if (threadIdx.x == 0) {
        while (*done < round) __nanosleep(100);
        __threadfence_system();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = __ldcv(y + i);   // uncached: the host wrote it after this kernel may have started
        float4 o = out[i];
        o.x += v.x; o.y += v.y; o.z += v.z; o.w += v.w;
        out[i] = o;
    }
}

}  // namespace

ExpertDoorbell::ExpertDoorbell(int max_m, int topk, int dim) : max_m_(max_m), topk_(topk), dim_(dim) {
    if (max_m < 1 || topk < 1 || dim < 8 || dim % 8 != 0) throw std::invalid_argument("ExpertDoorbell: bad shape");
    const size_t sel = (size_t) max_m * topk, rows = (size_t) max_m * dim;
    const size_t o_seq = 0, o_done = kAlign, o_x = 2 * kAlign;
    const size_t o_ids = o_x + up(rows * 2), o_w = o_ids + up(sel * 4), o_y = o_w + up(sel * 4);
    const size_t o_counts = o_y + up(rows * 4);
    const size_t total = o_counts + up(sizeof(Counts));
    ck(cudaHostAlloc(&host_, total, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(host_, 0, total);
    void* dev = nullptr;
    ck(cudaHostGetDevicePointer(&dev, host_, 0), "cudaHostGetDevicePointer");
    auto hp = [&](size_t o) { return (char*) host_ + o; };
    auto dp = [&](size_t o) { return (char*) dev + o; };
    h_seq_ = (uint32_t*) hp(o_seq);    d_seq_ = (uint32_t*) dp(o_seq);
    h_done_ = (uint32_t*) hp(o_done);  d_done_ = (uint32_t*) dp(o_done);
    h_x_ = (uint16_t*) hp(o_x);        d_x_ = (uint16_t*) dp(o_x);
    h_ids_ = (int32_t*) hp(o_ids);     d_ids_ = (int32_t*) dp(o_ids);
    h_w_ = (float*) hp(o_w);           d_w_ = (float*) dp(o_w);
    h_y_ = (float*) hp(o_y);           d_y_ = (float*) dp(o_y);
    h_counts_ = (Counts*) hp(o_counts); d_counts_ = (Counts*) dp(o_counts);
    try {
        ck(cudaMalloc(&gpu_experts_, sel * sizeof(*gpu_experts_)), "call descriptors");
    } catch (...) {
        cudaFreeHost(host_);
        throw;
    }
}

ExpertDoorbell::~ExpertDoorbell() {
    cudaFree(gpu_experts_);
    if (host_) cudaFreeHost(host_);
}

void ExpertDoorbell::publish(const uint16_t* x, const int32_t* ids, const float* w, int m, const int32_t* res,
                             int32_t* gpu_sel, uint32_t round, cudaStream_t stream,
                             const kernels::Exl3Expert* vram, const kernels::Exl3Expert* ram, const int* quota,
                             ExpertStaging* stage, const ExpertBlob* blobs) {
    if (m < 1 || m > max_m_) throw std::invalid_argument("ExpertDoorbell::publish: bad m");
    if (res && ram && !vram) throw std::invalid_argument("ExpertDoorbell::publish: missing VRAM descriptors");
    if (stage && (!blobs || stage->capacity() < m * topk_))
        throw std::invalid_argument("ExpertDoorbell::publish: missing blob metadata or staging slots");
    publish_k<<<1, 512, 0, stream>>>((const uint4*) x, m * dim_ / 8, ids, w, m, topk_, res, gpu_sel, (uint4*) d_x_,
                                     d_ids_, d_w_, d_seq_, round, vram, ram, quota, gpu_experts_, d_counts_,
                                     blobs, stage ? stage->data() : nullptr, stage ? stage->stride() : 0,
                                     stage ? stage->jobs() : nullptr, stage ? stage->count() : nullptr);
    ck(cudaGetLastError(), "publish");
}

void ExpertDoorbell::wait_add(float* out, int m, uint32_t round, cudaStream_t stream) {
    if (m < 1 || m > max_m_) throw std::invalid_argument("ExpertDoorbell::wait_add: bad m");
    wait_add_k<<<1, 1024, 0, stream>>>(d_done_, round, (const float4*) d_y_, (float4*) out, m * dim_ / 4);
    ck(cudaGetLastError(), "wait_add");
}

void ExpertDoorbell::reset() {
    __atomic_store_n(h_seq_, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(h_done_, 0u, __ATOMIC_RELEASE);
}

bool ExpertDoorbell::wait_published(uint32_t round, const std::atomic<bool>& stop) const {
    while (__atomic_load_n(h_seq_, __ATOMIC_ACQUIRE) < round) {
        if (stop.load(std::memory_order_relaxed)) return false;
        cpu_relax();
    }
    return true;
}

void ExpertDoorbell::mark_done(uint32_t round) { __atomic_store_n(h_done_, round, __ATOMIC_RELEASE); }

}  // namespace strata::ds41
