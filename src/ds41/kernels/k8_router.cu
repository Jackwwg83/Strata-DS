// K8-04: exact-tree, cross-CTA split-K with a GPU-only final selection.
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/config.hpp"
#include "k8/split_math.hpp"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::ds41::kernels {
namespace {
constexpr int kMaxTokens = 8;  // The fixed interface's complete legal range.
constexpr int kParts = 2;
constexpr int kPartLanes = 16;
constexpr int kExpertsPerBlock = 2;
constexpr int kThreads = kPartLanes * kExpertsPerBlock;
static_assert(kDim == 5120 && kExperts == 384 && kTopK == 6);

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "K8 split router: %s: %s\n", operation, cudaGetErrorString(error));
        std::abort();
    }
}

float* scratch_for_device() {
    struct Buffer { int device; float* partials; };
    static std::mutex mutex;
    static std::vector<Buffer> buffers;
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& buffer : buffers) {
        if (buffer.device == device) return buffer.partials;
    }
    float* partials = nullptr;
    // One K8-owned allocation per device, bounded by m<=8: 24 KiB. The
    // mandated eager call initializes it before graph capture. Never freed.
    // Same-task calls/replays on one device do not overlap; other tasks use
    // their own buffers and can execute concurrently on different streams.
    check(cudaMalloc(&partials, kMaxTokens * kParts * kExperts * sizeof(float)), "initialize scratch");
    buffers.push_back({device, partials});
    return partials;
}

template <int Tokens>
__global__ void split_dot(const __nv_bfloat16* __restrict__ x,
                          const __nv_bfloat16* __restrict__ w,
                          float* __restrict__ partials) {
    const int local_lane = threadIdx.x & (kPartLanes - 1);
    const int expert = blockIdx.x * kExpertsPerBlock + threadIdx.x / kPartLanes;
    const int part = blockIdx.y;
    const int reference_lane = 2 * local_lane + part;
    float sums[Tokens] = {};
    // Partition K by parity, not by contiguous ranges. Each lane retains
    // precisely its reference FMA chain (d=reference_lane, +32, ...).
    // The two CTAs compute the even and odd children of the reference's
    // final reduction node. This increases the GEMV grid from 48 baseline
    // CTAs (96 in K8-02 decode) to 384 without reassociating any additions.
#pragma unroll 8
    for (int d = reference_lane; d < kDim; d += 32) {
        const float weight = __bfloat162float(w[expert * kDim + d]);
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            sums[t] = __fmaf_rn(__bfloat162float(x[t * kDim + d]), weight, sums[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        // Logical offsets 8,4,2,1 are reference lane offsets 16,8,4,2.
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            sums[t] = __fadd_rn(sums[t], __shfl_down_sync(0xffffffffu, sums[t], offset, kPartLanes));
        }
        if (local_lane == 0) partials[(t * kParts + part) * kExperts + expert] = sums[t];
    }
}

__device__ __forceinline__ void warp_best(double& score, int& expert) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const double other_score = __shfl_down_sync(0xffffffffu, score, offset);
        const int other_expert = __shfl_down_sync(0xffffffffu, expert, offset);
        if (k8_split::better(other_score, other_expert, score, expert)) {
            score = other_score;
            expert = other_expert;
        }
    }
}

__global__ void finish_router(const float* __restrict__ partials,
                              const float* __restrict__ bias,
                              int32_t* __restrict__ ids,
                              float* __restrict__ weights) {
    constexpr int kWarps = kExperts / 32;
    __shared__ double warp_score[kWarps];
    __shared__ int warp_expert[kWarps];
    __shared__ int winner;
    __shared__ double selected[kTopK];
    const int token = blockIdx.x;
    const int expert = threadIdx.x;
    const int lane = expert & 31;
    const int warp = expert >> 5;
    // The original warp reduction's last offset=1 addition, bit for bit.
    const float logit = __fadd_rn(partials[(token * kParts) * kExperts + expert],
                                partials[(token * kParts + 1) * kExperts + expert]);
    const double raw = k8_split::score(logit);
    double value = raw + double(bias[expert]);
    int candidate = expert;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        double best = value;
        int best_id = candidate;
        warp_best(best, best_id);
        if (lane == 0) {
            warp_score[warp] = best;
            warp_expert[warp] = best_id;
        }
        __syncthreads();
        if (warp == 0) {
            best = lane < kWarps ? warp_score[lane] : -INFINITY;
            best_id = lane < kWarps ? warp_expert[lane] : kExperts;
            warp_best(best, best_id);
            if (lane == 0) winner = best_id;
        }
        __syncthreads();
        if (expert == winner) {
            ids[token * kTopK + rank] = expert;
            selected[rank] = raw;
            value = -INFINITY;
            candidate = kExperts;
        }
        // The next iteration's barrier also orders selected[] stores.
    }
    __syncthreads();
    if (expert == 0) {
        double sum = 0.0;
#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) sum += selected[rank];
#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
            weights[token * kTopK + rank] = float(selected[rank] / (sum + 1e-20) * double(kRouteScale));
        }
    }
}

template <int Tokens>
void launch(const __nv_bfloat16* x, const __nv_bfloat16* w, float* partials, cudaStream_t stream) {
    split_dot<Tokens><<<dim3(kExperts / kExpertsPerBlock, kParts), kThreads, 0, stream>>>(x, w, partials);
}
}  // namespace

void router_topk(const __nv_bfloat16* x, int m, const __nv_bfloat16* w, const float* bias,
                 int32_t* ids, float* weights, cudaStream_t stream) {
    if (m < 1 || m > kMaxTokens) return;
    float* partials = scratch_for_device();
    switch (m) {
        case 1: launch<1>(x, w, partials, stream); break;
        case 2: launch<2>(x, w, partials, stream); break;
        case 3: launch<3>(x, w, partials, stream); break;
        case 4: launch<4>(x, w, partials, stream); break;
        case 5: launch<5>(x, w, partials, stream); break;
        case 6: launch<6>(x, w, partials, stream); break;
        case 7: launch<7>(x, w, partials, stream); break;
        case 8: launch<8>(x, w, partials, stream); break;
    }
    // Supplied-stream ordering replaces any host copies/synchronization.
    finish_router<<<m, kExperts, 0, stream>>>(partials, bias, ids, weights);
    check(cudaGetLastError(), "launch");
}
}  // namespace strata::ds41::kernels
