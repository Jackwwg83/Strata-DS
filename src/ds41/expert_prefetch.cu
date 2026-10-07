// src/ds41/expert_prefetch.cu - see include/strata/ds41/expert_prefetch.hpp
#include "strata/ds41/expert_prefetch.hpp"

#include "strata/ds41/ops.hpp"

#include <stdexcept>
#include <string>

namespace strata::ds41 {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 prefetch: ") + what + ": " + cudaGetErrorString(e));
}

__device__ void rebase(kernels::Exl3Proj& p, const uint8_t* src, uint8_t* dst) {
    p.trellis = reinterpret_cast<const uint16_t*>(dst + (reinterpret_cast<const uint8_t*>(p.trellis) - src));
    p.suh = reinterpret_cast<const __half*>(dst + (reinterpret_cast<const uint8_t*>(p.suh) - src));
    p.svh = reinterpret_cast<const __half*>(dst + (reinterpret_cast<const uint8_t*>(p.svh) - src));
}

/// One block of n_experts threads (<= 1024): the router's selection score (kernels::router_topk: s = sqrt(softplus),
/// ranked by s + bias, ties to the lower id), the `g` best in order, then the copy plan.
__global__ void plan_k(const float* __restrict__ logits, const float* __restrict__ bias, int n, int g,
                       const int32_t* __restrict__ res, const kernels::Exl3Expert* __restrict__ ram,
                       const ExpertBlob* __restrict__ blobs, uint8_t* buffer, size_t cap, int32_t* ranked,
                       int32_t* ids, kernels::Exl3Expert* descs, ExpertCopy* jobs, int* count) {
    __shared__ float score[1024];
    const int e = threadIdx.x;
    if (e < n) {
        const float v = logits[e];
        const float sp = v > 20.0f ? v : log1pf(expf(v));
        score[e] = sqrtf(sp) + bias[e];
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    size_t used = 0;
    int k = 0;
    for (int r = 0; r < g; ++r) {
        int best = -1;
        for (int j = 0; j < n; ++j)
            if (score[j] > -INFINITY && (best < 0 || score[j] > score[best])) best = j;
        ranked[r] = best;
        if (best < 0) continue;
        score[best] = -INFINITY;
        if (res && res[best] >= 0) continue;            // in VRAM: no copy
        const kernels::Exl3Expert d = ram[best];
        if (!d.w1.trellis) continue;                   // not in the RAM tier (or on its way between tiers)
        const ExpertBlob blob = blobs[best];
        const size_t at = (used + 255) / 256 * 256;
        if (at + blob.bytes > cap) continue;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(d.w1.trellis) - blob.first_trellis;
        uint8_t* dst = buffer + at;
        jobs[k] = {src, dst, blob.bytes};
        kernels::Exl3Expert moved = d;
        rebase(moved.w1, src, dst);
        rebase(moved.w3, src, dst);
        rebase(moved.w2, src, dst);
        descs[k] = moved;
        ids[k++] = best;
        used = at + blob.bytes;
    }
    for (int j = k; j < g; ++j) ids[j] = -1;
    *count = k;
}

/// The planned copies, every block over every job (as the doorbell's staging copy)
__global__ void copy_k(const ExpertCopy* jobs, const int* count) {
    const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t step = size_t(gridDim.x) * blockDim.x;
    const int n = *count;
    for (int j = 0; j < n; ++j) {
        const ExpertCopy job = jobs[j];
        const auto* src = reinterpret_cast<const uint4*>(job.src);
        auto* dst = reinterpret_cast<uint4*>(job.dst);
        for (size_t i = lane; i < job.bytes / 16; i += step) dst[i] = src[i];
        const size_t tail = job.bytes / 16 * 16 + lane;
        if (tail < job.bytes) job.dst[tail] = job.src[tail];
    }
}

}  // namespace

ExpertPrefetch::ExpertPrefetch(int guesses, size_t buffer_bytes, int n_experts, int dim)
    : guesses_(guesses), n_experts_(n_experts), dim_(dim), buffer_bytes_(buffer_bytes) {
    if (guesses < 1 || guesses > kMaxGuesses || n_experts < 1 || n_experts > 1024 || buffer_bytes == 0)
        throw std::invalid_argument("ExpertPrefetch: bad shape");
    try {
        ck(cudaMalloc(&buffer_, 2 * buffer_bytes_), "buffer");
        ck(cudaMalloc(&logits_, n_experts_ * sizeof(float)), "logits");
        ck(cudaMalloc(&ids_, 2 * kMaxGuesses * sizeof(int32_t)), "ids");
        ck(cudaMalloc(&ranked_, 2 * kMaxGuesses * sizeof(int32_t)), "ranked");
        ck(cudaMalloc(&descs_, 2 * kMaxGuesses * sizeof(kernels::Exl3Expert)), "descriptors");
        ck(cudaMalloc(&jobs_, 2 * kMaxGuesses * sizeof(ExpertCopy)), "jobs");
        ck(cudaMalloc(&count_, 2 * sizeof(int)), "count");
        ck(cudaMemset(ids_, 0xff, 2 * kMaxGuesses * sizeof(int32_t)), "ids");
        ck(cudaMemset(count_, 0, 2 * sizeof(int)), "count");
        ck(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking), "stream");
        for (int p = 0; p < 2; ++p) {
            ck(cudaEventCreateWithFlags(&ready_[p], cudaEventDisableTiming), "event");
            ck(cudaEventCreateWithFlags(&done_[p], cudaEventDisableTiming), "event");
        }
    } catch (...) {
        this->~ExpertPrefetch();
        throw;
    }
}

ExpertPrefetch::~ExpertPrefetch() {
    if (copy_) cudaStreamSynchronize(copy_);
    for (int p = 0; p < 2; ++p) {
        if (ready_[p]) cudaEventDestroy(ready_[p]);
        if (done_[p]) cudaEventDestroy(done_[p]);
        ready_[p] = done_[p] = nullptr;
    }
    if (copy_) cudaStreamDestroy(copy_);
    copy_ = nullptr;
    for (void* p : {(void*) buffer_, (void*) logits_, (void*) ids_, (void*) ranked_, (void*) descs_, (void*) jobs_,
                    (void*) count_})
        if (p) cudaFree(p);
    buffer_ = nullptr;
    logits_ = nullptr;
    ids_ = ranked_ = nullptr;
    descs_ = nullptr;
    jobs_ = nullptr;
    count_ = nullptr;
}

void ExpertPrefetch::plan(int layer, const __nv_bfloat16* x, const __nv_bfloat16* w, const float* bias,
                          const int32_t* res, const kernels::Exl3Expert* ram, const ExpertBlob* blobs,
                          cudaStream_t main) {
    const int p = layer & 1;
    ops::bf16_linear(x, nullptr, w, dim_, n_experts_, nullptr, logits_, main);
    plan_k<<<1, n_experts_, 0, main>>>(logits_, bias, n_experts_, guesses_, res, ram, blobs,
                                       buffer_ + (size_t) p * buffer_bytes_, buffer_bytes_,
                                       ranked_ + p * kMaxGuesses, ids_ + p * kMaxGuesses,
                                       descs_ + p * kMaxGuesses, jobs_ + p * kMaxGuesses, count_ + p);
    ck(cudaGetLastError(), "plan launch");
}

void ExpertPrefetch::copy(int layer, cudaStream_t main) {
    const int p = layer & 1;
    ck(cudaEventRecord(ready_[p], main), "record");
    ck(cudaStreamWaitEvent(copy_, ready_[p], 0), "wait");
    copy_k<<<68, 256, 0, copy_>>>(jobs_ + p * kMaxGuesses, count_ + p);
    ck(cudaGetLastError(), "copy launch");
    ck(cudaEventRecord(done_[p], copy_), "record copy");
}

void ExpertPrefetch::join(int layer, cudaStream_t main) {
    ck(cudaStreamWaitEvent(main, done_[layer & 1], 0), "join");
}

}  // namespace strata::ds41
