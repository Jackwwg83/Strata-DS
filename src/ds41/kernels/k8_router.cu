// K8-08: full shared BF16 activations reused by four experts; streamed weights.
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/config.hpp"
#include "k8/math.hpp"
#include "k8/layout.hpp"

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

void initialize_shared_limits();

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
    // Opt in every legal shape before any capture, even when warmup uses m=1.
    initialize_shared_limits();
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

template <int Tokens>
__global__ void activation_reuse_scores(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ w,
                                        double* __restrict__ scores) {
    constexpr int Experts = k8_detail::kExpertsPerBlock;
    constexpr int Width = k8_detail::kWeightTile;
    static_assert(kDim % kWarp == 0 && Width % kWarp == 0);
    static_assert(k8_detail::shared_bytes(Tokens) <= 99 * 1024);
    extern __shared__ __align__(16) unsigned char storage[];
    auto* cached_x = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* tile_w = cached_x + Tokens * kDim;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int token = warp % Tokens;
    const int local_expert = warp / Tokens;
    const int expert = blockIdx.x * Experts + local_expert;

    // Every BF16 activation has one producer for this CTA. Its full lifetime
    // spans all weight phases and all four experts, without any reread from x.
    // Scalar loads need only the interface's ordinary BF16 source alignment.
    for (int i = threadIdx.x; i < Tokens * kDim; i += blockDim.x) cached_x[i] = x[i];
    float acc = 0.0f;
    for (int base = 0; base < kDim; base += Width) {
        const int count = kDim - base < Width ? kDim - base : Width;
        // Each real weight has one producer, independent of the token count.
        // Padding in the last 1,024-column phase is neither loaded nor read.
        for (int i = threadIdx.x; i < Experts * Width; i += blockDim.x) {
            const int row = i / Width;
            const int d = i % Width;
            if (d < count) tile_w[i] = w[(blockIdx.x * Experts + row) * kDim + base + d];
        }
        // Publishes the initial full x cache as well as this weight phase.
        __syncthreads();
#pragma unroll 8
        for (int d = lane; d < count; d += kWarp) {
            const float xv = __bfloat162float(cached_x[token * kDim + base + d]);
            const float wv = __bfloat162float(tile_w[local_expert * Width + d]);
            // Carry the same accumulator through every phase: no split-K
            // partial sums and no change to the reference's lane FMA chain.
            acc = __fmaf_rn(xv, wv, acc);
        }
        // Retire every weight consumer before any producer overwrites tile_w.
        __syncthreads();
    }
    acc = warp_sum(acc);
    if (lane == 0) scores[token * kExperts + expert] = k8_detail::score(acc);
}

template <int Tokens>
void initialize_shared_limit() {
    constexpr int bytes = k8_detail::shared_bytes(Tokens);
    if constexpr (bytes > 48 * 1024) {
        check(cudaFuncSetAttribute(activation_reuse_scores<Tokens>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes),
              "initialize shared memory limit");
    }
}

void initialize_shared_limits() {
    initialize_shared_limit<1>();
    initialize_shared_limit<2>();
    initialize_shared_limit<3>();
    initialize_shared_limit<4>();
    initialize_shared_limit<5>();
    initialize_shared_limit<6>();
    initialize_shared_limit<7>();
    initialize_shared_limit<8>();
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
    constexpr int threads = k8_detail::kExpertsPerBlock * Tokens * kWarp;
    constexpr int bytes = k8_detail::shared_bytes(Tokens);
    activation_reuse_scores<Tokens><<<kExperts / k8_detail::kExpertsPerBlock, threads, bytes, stream>>>(x, w, scores);
}

}  // namespace

void router_topk(const __nv_bfloat16* x, int m, const __nv_bfloat16* w, const float* bias,
                 int32_t* ids, float* weights, cudaStream_t stream) {
    if (m < 1 || m > kMaxTokens) return;
    double* scores = workspace();
    switch (m) {
        case 1: launch_tile<1>(x, w, scores, stream); break;
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
