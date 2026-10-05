// K8: one weight read for the token tile, followed by a GPU-only stable top six.
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

__device__ __forceinline__ float warp_sum(float value) {
    // Identical lane ownership, FMA order, and reduction tree to bf16_gemv_k
    // in ops.cu. In particular, do not split K or reassociate partial sums.
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__global__ void decode_scores(const __nv_bfloat16* __restrict__ x,
                              const __nv_bfloat16* __restrict__ w,
                              double* __restrict__ scores) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * 4 + (threadIdx.x >> 5);
    const __nv_bfloat16* row = w + expert * kDim;
    float acc = 0.0f;
#pragma unroll 8
    for (int d = lane; d < kDim; d += kWarp) {
        acc = __fmaf_rn(__bfloat162float(x[d]), __bfloat162float(row[d]), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) scores[expert] = k8_detail::score(acc);
}

template <int Tokens>
__global__ void tile_scores(const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ w,
                            double* __restrict__ scores) {
    // Each CTA owns one expert. Cooperatively load and convert its entire
    // BF16 row once, then reuse it across all token warps. Compared with one
    // warp holding eight accumulators, this exposes independent token warps
    // without changing any token's FP32 accumulation sequence.
    __shared__ float row[kDim];
    const int expert = blockIdx.x;
    for (int d = threadIdx.x; d < kDim; d += blockDim.x) {
        row[d] = __bfloat162float(w[expert * kDim + d]);
    }
    __syncthreads();
    const int token = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (token < Tokens) {
        float acc = 0.0f;
#pragma unroll 8
        for (int d = lane; d < kDim; d += kWarp) {
            acc = __fmaf_rn(__bfloat162float(x[token * kDim + d]), row[d], acc);
        }
        acc = warp_sum(acc);
        // Scores are computed where logits are produced, spreading the small
        // nonlinear step across the GEMV grid instead of bottlenecking one CTA.
        if (lane == 0) scores[token * kExperts + expert] = k8_detail::score(acc);
    }
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
void launch_tile(const __nv_bfloat16* x, const __nv_bfloat16* w, double* scores, cudaStream_t stream) {
    constexpr int threads = Tokens <= 4 ? 128 : 256;
    tile_scores<Tokens><<<kExperts, threads, 0, stream>>>(x, w, scores);
}

}  // namespace

void router_topk(const __nv_bfloat16* x, int m, const __nv_bfloat16* w, const float* bias,
                 int32_t* ids, float* weights, cudaStream_t stream) {
    if (m < 1 || m > kMaxTokens) return;
    double* scores = workspace();
    switch (m) {
        case 1: decode_scores<<<kExperts / 4, 128, 0, stream>>>(x, w, scores); break;
        case 2: launch_tile<2>(x, w, scores, stream); break;
        case 3: launch_tile<3>(x, w, scores, stream); break;
        case 4: launch_tile<4>(x, w, scores, stream); break;
        case 5: launch_tile<5>(x, w, scores, stream); break;
        case 6: launch_tile<6>(x, w, scores, stream); break;
        case 7: launch_tile<7>(x, w, scores, stream); break;
        case 8: launch_tile<8>(x, w, scores, stream); break;
    }
    select_top6<<<m, kSelectThreads, 0, stream>>>(scores, bias, ids, weights);
    check(cudaGetLastError(), "launch");
}

}  // namespace strata::ds41::kernels
