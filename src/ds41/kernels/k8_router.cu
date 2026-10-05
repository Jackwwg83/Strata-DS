// K8-05: 8-expert producer groups -> deterministic local top-6 -> GPU merge.
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/config.hpp"

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace strata::ds41::kernels {
namespace {

constexpr int kMaxTokens = 8;
constexpr int kGroupExperts = 8;
constexpr int kGroups = kExperts / kGroupExperts;
constexpr int kCandidates = kGroups * kTopK;
constexpr int kThreads = 256;
constexpr unsigned kFullWarp = 0xffffffffu;
static_assert(kExperts == 384 && kDim == 5120 && kTopK == 6);
static_assert(kExperts % kGroupExperts == 0 && kGroupExperts == kThreads / 32);

struct Candidate {
    double biased;
    double unbiased;
    int id;
};
static_assert(sizeof(Candidate) == 24);

__device__ __forceinline__ Candidate empty_candidate() {
    return {-INFINITY, 0.0, INT_MAX};
}

// Both levels use exactly the same total order (including equal scores).
__device__ __forceinline__ bool precedes(const Candidate& a, const Candidate& b) {
    return a.biased > b.biased || (a.biased == b.biased && a.id < b.id);
}

__device__ __forceinline__ Candidate warp_best(Candidate a) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        Candidate b;
        b.biased = __shfl_down_sync(kFullWarp, a.biased, offset);
        b.unbiased = __shfl_down_sync(kFullWarp, a.unbiased, offset);
        b.id = __shfl_down_sync(kFullWarp, a.id, offset);
        if (precedes(b, a)) a = b;
    }
    return a;  // The full reduction is in lane zero.
}

__device__ __forceinline__ double score(float logit) {
    // The fixed acceptance reference evaluates the nonlinear function in double.
    // Do not round s+b to FP32: it can create artificial ties in the routing order.
    const double z = static_cast<double>(logit);
    return sqrt(z > 20.0 ? z : log1p(exp(z)));
}

template <int M>
__global__ void produce(const __nv_bfloat16* __restrict__ x,
                        const __nv_bfloat16* __restrict__ w,
                        const float* __restrict__ bias,
                        Candidate* __restrict__ candidates) {
    __shared__ double raw[M][kGroupExperts];
    __shared__ double ranked[M][kGroupExperts];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int group = blockIdx.x;
    const int expert = group * kGroupExperts + warp;
    float acc[M] = {};

    // One warp owns one complete expert row. A weight is loaded once, then
    // reused for every token. For each token, keep the reference's 160 ordered
    // FP32 lane FMAs at i=lane+32*j; no split-K or reassociated dot product.
#pragma unroll 1
    for (int i = lane; i < kDim; i += 32) {
        const float wi = __bfloat162float(w[expert * kDim + i]);
#pragma unroll
        for (int t = 0; t < M; ++t) {
            acc[t] = __fmaf_rn(__bfloat162float(x[t * kDim + i]), wi, acc[t]);
        }
    }

    // Preserve bf16_gemv_k's 16,8,4,2,1 shuffle-down sum. Scatter the resulting
    // logits to lanes 0..M-1 so the double nonlinear work runs across tokens.
    float logit = 0.0f;
#pragma unroll
    for (int t = 0; t < M; ++t) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[t] = __fadd_rn(acc[t], __shfl_down_sync(kFullWarp, acc[t], offset));
        }
        const float z = __shfl_sync(kFullWarp, acc[t], 0);
        if (lane == t) logit = z;
    }
    if (lane < M) {
        const double s = score(logit);
        raw[lane][warp] = s;
        ranked[lane][warp] = s + static_cast<double>(bias[expert]);
    }
    __syncthreads();

    // Retask one warp per token. Only eight score triples are ever live here;
    // the producer writes six winners, never the full 384-expert score matrix.
    if (warp < M) {
        Candidate mine = empty_candidate();
        if (lane < kGroupExperts) {
            mine = {ranked[warp][lane], raw[warp][lane], group * kGroupExperts + lane};
        }
#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
            const Candidate winner = warp_best(mine);
            const int winner_id = __shfl_sync(kFullWarp, winner.id, 0);
            if (lane == 0) {
                candidates[(warp * kGroups + group) * kTopK + rank] = winner;
            }
            if (mine.id == winner_id) mine = empty_candidate();
        }
    }
}

__global__ void merge(const Candidate* __restrict__ candidates,
                      int32_t* __restrict__ ids, float* __restrict__ weights) {
    __shared__ Candidate warp_winners[kThreads / 32];
    __shared__ Candidate winner;
    __shared__ double selected[kTopK];
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int warp = thread >> 5;
    const int token = blockIdx.x;
    const Candidate* row = candidates + token * kCandidates;
    Candidate a = row[thread];  // 256 < 288 candidates: every first slot is valid.
    Candidate b = empty_candidate();
    if (thread + kThreads < kCandidates) b = row[thread + kThreads];

#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        Candidate best = warp_best(precedes(b, a) ? b : a);
        if (lane == 0) warp_winners[warp] = best;
        __syncthreads();
        if (warp == 0) {
            best = lane < kThreads / 32 ? warp_winners[lane] : empty_candidate();
            best = warp_best(best);
            if (lane == 0) {
                winner = best;
                selected[rank] = best.unbiased;
                ids[token * kTopK + rank] = best.id;
            }
        }
        __syncthreads();
        const int winner_id = winner.id;
        if (a.id == winner_id) a = empty_candidate();
        if (b.id == winner_id) b = empty_candidate();
    }
    if (thread == 0) {
        // Sum the original, unbiased s values in final routing order.
        double sum = 0.0;
#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) sum += selected[rank];
#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
            weights[token * kTopK + rank] =
                static_cast<float>(selected[rank] / (sum + 1e-20) * 1.5);
        }
    }
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "K8-05 %s: %s\n", operation, cudaGetErrorString(error));
        std::abort();
    }
}

Candidate* task_scratch() {
    // The engine serializes this task on each device, including graph replay.
    // Other tasks may overlap: this arena belongs exclusively to router_topk.
    // Allocate the full m<=8 capacity once per device, even when warmed at m=1.
    // The map has no artificial device-count cap; find() does not allocate.
    static std::mutex mutex;
    static std::map<int, Candidate*> per_device;
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    std::lock_guard<std::mutex> lock(mutex);
    const auto existing = per_device.find(device);
    if (existing != per_device.end()) return existing->second;
    Candidate* memory = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&memory),
                     sizeof(Candidate) * kMaxTokens * kCandidates), "allocate first-call scratch");
    per_device.emplace(device, memory);
    return memory;  // Intentionally retained for the process lifetime.
}

template <int M>
void launch(const __nv_bfloat16* x, const __nv_bfloat16* w, const float* bias,
            Candidate* candidates, cudaStream_t stream) {
    produce<M><<<kGroups, kThreads, 0, stream>>>(x, w, bias, candidates);
}

}  // namespace

void router_topk(const __nv_bfloat16* x, int m, const __nv_bfloat16* w, const float* bias,
                 int32_t* ids, float* weights, cudaStream_t stream) {
    if (m < 1 || m > kMaxTokens) {
        std::fprintf(stderr, "K8-05 router_topk: m must be in [1,8]\n");
        std::abort();
    }
    Candidate* candidates = task_scratch();
    switch (m) {
        case 1: launch<1>(x, w, bias, candidates, stream); break;
        case 2: launch<2>(x, w, bias, candidates, stream); break;
        case 3: launch<3>(x, w, bias, candidates, stream); break;
        case 4: launch<4>(x, w, bias, candidates, stream); break;
        case 5: launch<5>(x, w, bias, candidates, stream); break;
        case 6: launch<6>(x, w, bias, candidates, stream); break;
        case 7: launch<7>(x, w, bias, candidates, stream); break;
        case 8: launch<8>(x, w, bias, candidates, stream); break;
    }
    check(cudaGetLastError(), "produce launch");
    merge<<<m, kThreads, 0, stream>>>(candidates, ids, weights);
    check(cudaGetLastError(), "merge launch");
}

}  // namespace strata::ds41::kernels
