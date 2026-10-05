// K8: last-completing-CTA decode; unchanged two-stage token-tile routing.
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/config.hpp"
#include "k8/math.hpp"

#include <cuda/atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::ds41::kernels {
namespace {

constexpr int kMaxTokens = 8;  // Fixed interface, not a benchmark-derived limit.
constexpr int kWarp = 32;
constexpr int kSelectThreads = kWarp;  // One independent warp CTA per token.
static_assert(kDim == 5120 && kExperts == 384 && kTopK == 6);

void check(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "ds41 router: %s: %s\n", where, cudaGetErrorString(error));
        std::abort();
    }
}

using Completion = cuda::atomic_ref<unsigned, cuda::thread_scope_device>;
struct Workspace {
    double scores[kMaxTokens * kExperts];
    alignas(Completion::required_alignment) unsigned completed;
};
static_assert(offsetof(Workspace, completed) % Completion::required_alignment == 0);
static_assert(sizeof(Workspace) == 24584);

Workspace* workspace(cudaStream_t stream) {
    // One K8-owned, maximum-size arena per device, retained for graph replay.
    // The engine orders K8 calls/replays without overlap on a device, including
    // across streams. Other task kernels own disjoint scratch and may overlap.
    struct Entry { int device; Workspace* arena; };
    static std::mutex mutex;
    static std::vector<Entry> buffers;
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    std::lock_guard<std::mutex> lock(mutex);
    for (const Entry& buffer : buffers) {
        if (buffer.device == device) return buffer.arena;
    }
    // All legal first eager shapes initialize the counter, including m=2..8.
    // Capture only reuses the arena; it cannot initialize or enlarge it.
    cudaStreamCaptureStatus capture_status;
    check(cudaStreamIsCapturing(stream, &capture_status), "query initial capture");
    if (capture_status != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "ds41 router: one eager call is required before capture\n");
        std::abort();
    }
    Workspace* arena = nullptr;
    check(cudaMalloc(&arena, sizeof(Workspace)), "initialize scratch");
    check(cudaMemsetAsync(&arena->completed, 0, sizeof(unsigned), stream), "initialize completion");
    buffers.push_back({device, arena});
    return arena;
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
    constexpr int kPerThread = kExperts / kWarp;
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    // Twelve register-resident candidates per lane, with the original expert
    // IDs retained for ties. Each token is an independent, full-warp CTA.
    double raw[kPerThread];
    double values[kPerThread];
    int expert_ids[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int id = lane + j * kWarp;
        raw[j] = scores[token * kExperts + id];
        values[j] = raw[j] + double(bias[id]);
        expert_ids[j] = id;
    }
    double selected = 0.0;
    double sum = 0.0;
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
        const int chosen = __shfl_sync(0xffffffffu, id, 0);
        double unbiased = 0.0;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            if (expert_ids[j] == chosen) {
                unbiased = raw[j];
                values[j] = -INFINITY;
                expert_ids[j] = kExperts;
            }
        }
        // Broadcast the owner's original score, including zero. Subtracting
        // the bias from a rounded comparison value would change normalization.
        unbiased = __shfl_sync(0xffffffffu, unbiased, chosen & (kWarp - 1));
        if (lane == 0) {
            ids[token * kTopK + i] = chosen;
            // Exactly the reference's left-to-right selected-score sum.
            sum += unbiased;
        }
        if (lane == i) selected = unbiased;
    }
    sum = __shfl_sync(0xffffffffu, sum, 0);
    if (lane < kTopK) {
        weights[token * kTopK + lane] = float(selected / (sum + 1e-20) * double(kRouteScale));
    }
}

__device__ __forceinline__ void select_decode_top6(const double* __restrict__ scores,
                            const float* __restrict__ bias,
                            int32_t* __restrict__ ids,
                            float* __restrict__ weights) {
    constexpr int kPerThread = kExperts / kWarp;
    constexpr int token = 0;
    const int lane = threadIdx.x;
    // Twelve register-resident candidates per lane, with the original expert
    // IDs retained for ties. Each token is an independent, full-warp CTA.
    double raw[kPerThread];
    double values[kPerThread];
    int expert_ids[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int id = lane + j * kWarp;
        raw[j] = scores[token * kExperts + id];
        values[j] = raw[j] + double(bias[id]);
        expert_ids[j] = id;
    }
    double selected = 0.0;
    double sum = 0.0;
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
        const int chosen = __shfl_sync(0xffffffffu, id, 0);
        double unbiased = 0.0;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            if (expert_ids[j] == chosen) {
                unbiased = raw[j];
                values[j] = -INFINITY;
                expert_ids[j] = kExperts;
            }
        }
        // Broadcast the owner's original score, including zero. Subtracting
        // the bias from a rounded comparison value would change normalization.
        unbiased = __shfl_sync(0xffffffffu, unbiased, chosen & (kWarp - 1));
        if (lane == 0) {
            ids[token * kTopK + i] = chosen;
            // Exactly the reference's left-to-right selected-score sum.
            sum += unbiased;
        }
        if (lane == i) selected = unbiased;
    }
    sum = __shfl_sync(0xffffffffu, sum, 0);
    if (lane < kTopK) {
        weights[token * kTopK + lane] = float(selected / (sum + 1e-20) * double(kRouteScale));
    }
}

__global__ void decode_top6(const __nv_bfloat16* __restrict__ x,
                              const __nv_bfloat16* __restrict__ w,
                              const float* __restrict__ bias,
                              int32_t* __restrict__ ids,
                              float* __restrict__ weights, Workspace* arena) {
    __shared__ bool last;
    double* scores = arena->scores;
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * 4 + (threadIdx.x >> 5);
    const __nv_bfloat16* row = w + expert * kDim;
    float acc = 0.0f;
#pragma unroll 8
    for (int d = lane; d < kDim; d += kWarp) {
        acc = __fmaf_rn(__bfloat162float(x[d]), __bfloat162float(row[d]), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        scores[expert] = k8_detail::score(acc);
        // Each actual score writer fences its own store before CTA publication,
        // following the documented threadFenceReduction producer pattern.
        __threadfence();
    }

    // Publish all four score writers through the leader's device-scope RMW.
    __syncthreads();
    if (threadIdx.x == 0) {
        Completion completed(arena->completed);
        // Each acq_rel RMW acquires and republishes its predecessor's history.
        // The last ticket therefore acquires all 384 scores transitively.
        const unsigned ticket = completed.fetch_add(1, cuda::memory_order_acq_rel);
        last = ticket == unsigned(kExperts / 4 - 1);
    }
    // Pass the acquired history and winner flag to the complete CTA.
    __syncthreads();
    if (!last) return;  // Uniform; no spinning or grid-residency assumption.
    // Every thread of the winning CTA fences after learning it is last. Thus
    // all selector lanes execute the fence before loading any global score.
    // Keep the acq_rel ticket and both publication barriers as well.
    __threadfence();
    if (threadIdx.x < kWarp) select_decode_top6(scores, bias, ids, weights);

    // All score reads and output writes finish before the reusable zero.
    // Every other CTA has issued its only counter operation. The next K8 call
    // is ordered after this kernel by the engine's same-task nonoverlap rule.
    __syncthreads();
    if (threadIdx.x == 0) Completion(arena->completed).store(0, cuda::memory_order_release);
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
    Workspace* arena = workspace(stream);
    double* scores = arena->scores;
    if (m == 1) {
        decode_top6<<<kExperts / 4, 128, 0, stream>>>(x, w, bias, ids, weights, arena);
        check(cudaGetLastError(), "launch decode");
        return;
    }
    switch (m) {
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
