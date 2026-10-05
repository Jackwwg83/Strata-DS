// K8-09: packed BF16 pair GEMV with all-token weight reuse, then GPU top six.
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/config.hpp"
#include "k8/math.hpp"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::ds41::kernels {
namespace {

constexpr int kMaxTokens = 8;  // Fixed interface, not a benchmark-derived limit.
constexpr int kWarp = 32;
constexpr int kSelectThreads = 128;
constexpr int kExpertThreads = 16;
constexpr int kScoreThreads = 32;
constexpr int kExpertsPerBlock = kScoreThreads / kExpertThreads;
static_assert(kDim == 5120 && kExperts == 384 && kTopK == 6);

void check(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "ds41 router: %s: %s\n", where, cudaGetErrorString(error));
        std::abort();
    }
}

struct Workspace {
    int device;
    double* scores;
};

double* workspace() {
    // Exactly one K8-owned, maximum-size allocation per device. The engine
    // warms up before capture and never overlaps two K8 calls on one device.
    // Other task kernels have independent storage and may run concurrently.
    static std::mutex mutex;
    static std::vector<Workspace> buffers;
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    std::lock_guard<std::mutex> lock(mutex);
    for (const Workspace& buffer : buffers) {
        if (buffer.device == device) return buffer.scores;
    }
    double* scores = nullptr;
    check(cudaMalloc(&scores, kMaxTokens * kExperts * sizeof(double)), "initialize scratch");
    buffers.push_back({device, scores});
    return scores;  // Kept for process lifetime, including captured graph replay.
}

// The fixed interface promises BF16 alignment, not four-byte alignment. The
// packed specialization is selected only when both complete row arrays allow it.
template <bool Aligned>
__device__ __forceinline__ __nv_bfloat162 load_pair(const __nv_bfloat16* p, int pair) {
    if constexpr (Aligned) return reinterpret_cast<const __nv_bfloat162*>(p)[pair];
    return __halves2bfloat162(p[2 * pair], p[2 * pair + 1]);
}

template <int Tokens, bool Aligned>
__global__ void packed_scores(const __nv_bfloat16* __restrict__ x,
                              const __nv_bfloat16* __restrict__ w,
                              double* __restrict__ scores) {
    static_assert(Tokens >= 1 && Tokens <= kMaxTokens);
    static_assert(kExperts % kExpertsPerBlock == 0 && kDim % 32 == 0);
    const int lane = threadIdx.x & (kExpertThreads - 1);
    const int expert = blockIdx.x * kExpertsPerBlock + threadIdx.x / kExpertThreads;
    const __nv_bfloat16* row = w + expert * kDim;
    float even[Tokens] = {};
    float odd[Tokens] = {};
    // Physical lane l owns reference lanes 2*l and 2*l+1. The single packed
    // weight read is reused by every token without changing either FMA chain:
    // d = 2*l + 32*j and d+1, with j increasing from 0 to 159.
#pragma unroll 4
    for (int pair = lane; pair < kDim / 2; pair += kExpertThreads) {
        const float2 weight = __bfloat1622float2(load_pair<Aligned>(row, pair));
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const float2 value = __bfloat1622float2(load_pair<Aligned>(x + token * kDim, pair));
            even[token] = __fmaf_rn(value.x, weight.x, even[token]);
            odd[token] = __fmaf_rn(value.y, weight.y, odd[token]);
        }
    }
    float own_logit = 0.0f;
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        // Reference offsets 16,8,4,2 act separately on the even and odd
        // lanes. In physical lane coordinates these are 8,4,2,1. Its final
        // offset 1 is exactly even[0]+odd[0], with the same operand order.
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            even[token] = __fadd_rn(even[token], __shfl_down_sync(0xffffffffu, even[token], off, kExpertThreads));
            odd[token] = __fadd_rn(odd[token], __shfl_down_sync(0xffffffffu, odd[token], off, kExpertThreads));
        }
        const float logit = __fadd_rn(even[token], odd[token]);
        const float leader_logit = __shfl_sync(0xffffffffu, logit, 0, kExpertThreads);
        if (lane == token) own_logit = leader_logit;
    }
    // Each token uses its own physical lane for the nonlinear step. All
    // shuffles above execute with complete warps and width-16 isolation.
    if (lane < Tokens) scores[lane * kExperts + expert] = k8_detail::score(own_logit);
}

__device__ __forceinline__ void warp_best(double& value, int& id) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const double other = __shfl_down_sync(0xffffffffu, value, offset);
        const int other_id = __shfl_down_sync(0xffffffffu, id, offset);
        if (k8_detail::better(other, other_id, value, id)) {
            value = other;
            id = other_id;
        }
    }
}

__global__ void select_top6(const double* __restrict__ scores,
                            const float* __restrict__ bias,
                            int32_t* __restrict__ ids,
                            float* __restrict__ weights) {
    constexpr int kWarps = kSelectThreads / kWarp;
    constexpr int kPerThread = kExperts / kSelectThreads;
    __shared__ double warp_values[kWarps];
    __shared__ int warp_ids[kWarps];
    __shared__ int winner;
    __shared__ double selected[kTopK];
    const int token = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    double raw[kPerThread];
    double values[kPerThread];
    int expert_ids[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int id = threadIdx.x + j * kSelectThreads;
        raw[j] = scores[token * kExperts + id];
        values[j] = raw[j] + double(bias[id]);
        expert_ids[j] = id;
    }
#pragma unroll
    for (int i = 0; i < kTopK; ++i) {
        double value = -INFINITY;
        int id = kExperts;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            if (k8_detail::better(values[j], expert_ids[j], value, id)) {
                value = values[j];
                id = expert_ids[j];
            }
        }
        warp_best(value, id);
        if (lane == 0) {
            warp_values[warp] = value;
            warp_ids[warp] = id;
        }
        __syncthreads();
        if (warp == 0) {
            value = lane < kWarps ? warp_values[lane] : -INFINITY;
            id = lane < kWarps ? warp_ids[lane] : kExperts;
            warp_best(value, id);
            if (lane == 0) winner = id;
        }
        __syncthreads();
        const int chosen = winner;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            if (expert_ids[j] == chosen) {
                // Exactly one thread owns each expert; unbiased zero scores
                // are valid and require no ballot or special owner inference.
                ids[token * kTopK + i] = chosen;
                selected[i] = raw[j];
                values[j] = -INFINITY;
                expert_ids[j] = kExperts;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double sum = 0.0;
#pragma unroll
        for (int i = 0; i < kTopK; ++i) sum += selected[i];
#pragma unroll
        for (int i = 0; i < kTopK; ++i) {
            weights[token * kTopK + i] = float(selected[i] / (sum + 1e-20) * double(kRouteScale));
        }
    }
}

template <int Tokens>
void launch_scores(const __nv_bfloat16* x, const __nv_bfloat16* w, double* scores, cudaStream_t stream) {
    const bool aligned = ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) & 3u) == 0;
    if (aligned) packed_scores<Tokens, true><<<kExperts / kExpertsPerBlock, kScoreThreads, 0, stream>>>(x, w, scores);
    else packed_scores<Tokens, false><<<kExperts / kExpertsPerBlock, kScoreThreads, 0, stream>>>(x, w, scores);
}

}  // namespace

void router_topk(const __nv_bfloat16* x, int m, const __nv_bfloat16* w, const float* bias,
                 int32_t* ids, float* weights, cudaStream_t stream) {
    if (m < 1 || m > kMaxTokens) return;
    double* scores = workspace();
    switch (m) {
        case 1: launch_scores<1>(x, w, scores, stream); break;
        case 2: launch_scores<2>(x, w, scores, stream); break;
        case 3: launch_scores<3>(x, w, scores, stream); break;
        case 4: launch_scores<4>(x, w, scores, stream); break;
        case 5: launch_scores<5>(x, w, scores, stream); break;
        case 6: launch_scores<6>(x, w, scores, stream); break;
        case 7: launch_scores<7>(x, w, scores, stream); break;
        case 8: launch_scores<8>(x, w, scores, stream); break;
    }
    select_top6<<<m, kSelectThreads, 0, stream>>>(scores, bias, ids, weights);
    check(cudaGetLastError(), "launch");
}

}  // namespace strata::ds41::kernels
