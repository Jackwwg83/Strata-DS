// K8-06: asynchronous double-buffered BF16 tiles, then deterministic GPU top six.
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/config.hpp"
#include "k8/math.hpp"
#include "k8/pipeline.hpp"

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

// All destination rows and chunks are 16-byte aligned. The fixed interface only
// promises BF16 source alignment; offset views may require the scalar fallback.
__device__ __forceinline__ void copy_bf16_chunk(__nv_bfloat16* dst, const __nv_bfloat16* src) {
#if __CUDA_ARCH__ >= 800
    if ((reinterpret_cast<uintptr_t>(src) & 15u) == 0) {
        const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src) : "memory");
        return;
    }
#endif
#pragma unroll
    for (int i = 0; i < k8_detail::kCopyValues; ++i) dst[i] = src[i];
}

__device__ __forceinline__ void commit_stage() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void wait_stage() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

template <int Tokens>
__device__ __forceinline__ void stage_tile(__nv_bfloat16* dst,
                                          const __nv_bfloat16* x,
                                          const __nv_bfloat16* w, int base) {
    constexpr int Width = k8_detail::kTile;
    constexpr int Experts = k8_detail::kExpertsPerBlock;
    constexpr int ChunksPerRow = Width / k8_detail::kCopyValues;
    constexpr int Chunks = (Experts + Tokens) * ChunksPerRow;
    // Unique 16-byte owner for every weight and input chunk; weight traffic is
    // independent of m, while the staged input tile is shared by both experts.
    for (int chunk = threadIdx.x; chunk < Chunks; chunk += blockDim.x) {
        const int row = chunk / ChunksPerRow;
        const int d = (chunk % ChunksPerRow) * k8_detail::kCopyValues;
        const __nv_bfloat16* src = row < Experts
            ? w + (blockIdx.x * Experts + row) * kDim + base + d
            : x + (row - Experts) * kDim + base + d;
        copy_bf16_chunk(dst + row * Width + d, src);
    }
    // Even threads with no copy in a stage commit and wait their own group.
    commit_stage();
}

template <int Tokens>
__global__ void pipeline_scores(const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ w,
                                double* __restrict__ scores) {
    constexpr int Width = k8_detail::kTile;
    constexpr int Experts = k8_detail::kExpertsPerBlock;
    constexpr int Rows = Experts + Tokens;
    static_assert(kDim % Width == 0 && Width % kWarp == 0);
    static_assert(2 * Rows * Width * sizeof(__nv_bfloat16) <= 99 * 1024);
    __shared__ __align__(16) __nv_bfloat16 tile[2][Rows][Width];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int local_expert = warp / Tokens;
    const int token = warp % Tokens;
    const int expert = blockIdx.x * Experts + local_expert;
    float acc = 0.0f;

    stage_tile<Tokens>(&tile[0][0][0], x, w, 0);
    wait_stage();
    __syncthreads();  // All producers' initial copies are visible to consumers.
    for (int base = 0, current = 0; base < kDim; base += Width, current ^= 1) {
        if (base + Width < kDim) {
            // Previous iteration's barrier retired every reader of this buffer.
            // This copy overlaps the other buffer's strictly ordered dot work.
            stage_tile<Tokens>(&tile[current ^ 1][0][0], x, w, base + Width);
        }
#pragma unroll 8
        for (int d = lane; d < Width; d += kWarp) {
            const float xv = __bfloat162float(tile[current][Experts + token][d]);
            const float wv = __bfloat162float(tile[current][local_expert][d]);
            acc = __fmaf_rn(xv, wv, acc);
        }
        wait_stage();
        // wait_group alone is per-thread: the CTA barrier both publishes the
        // next tile and prevents reuse of the current tile before every reader
        // finishes. No consumer can race a producer on either ping-pong buffer.
        __syncthreads();
    }
    acc = warp_sum(acc);
    if (lane == 0) scores[token * kExperts + expert] = k8_detail::score(acc);
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
    pipeline_scores<Tokens><<<kExperts / k8_detail::kExpertsPerBlock, threads, 0, stream>>>(x, w, scores);
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
