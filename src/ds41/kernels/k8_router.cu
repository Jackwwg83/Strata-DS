// One CTA per token: BF16 GEMV, deterministic top-6 and normalization.
// No device workspace, allocator, host readback, or inter-CTA synchronization.
#include "strata/ds41/kernels/k8_router.hpp"

#include <cmath>

namespace strata::ds41::kernels {
namespace {

constexpr int kDim = 5120;
constexpr int kExperts = 384;
constexpr int kTopK = 6;
constexpr int kThreads = 1024;
constexpr int kExpertThreads = 16;
constexpr int kExpertGroups = kThreads / kExpertThreads;
constexpr unsigned kWarpMask = 0xffffffffu;

// The acceptance reference applies this transform in double to FP32 GEMV logits.
// Compute it once for every expert; ranking never uses a rounded FP32 score.
__device__ __noinline__ double reference_score(float logit) {
    const double z = static_cast<double>(logit);
    return sqrt(z > 20.0 ? z : log1p(exp(z)));
}

__device__ __forceinline__ bool better(int a, int b, const double* biased) {
    if (a < 0) return false;
    if (b < 0) return true;
    return biased[a] > biased[b] || (biased[a] == biased[b] && a < b);
}

// Packed loads are an optimization, not an extra interface alignment contract.
template <bool Aligned>
__device__ __forceinline__ __nv_bfloat162 load_pair(const __nv_bfloat16* p, int pair) {
    if constexpr (Aligned) return reinterpret_cast<const __nv_bfloat162*>(p)[pair];
    return __halves2bfloat162(p[2 * pair], p[2 * pair + 1]);
}

template <bool Aligned>
__global__ __launch_bounds__(kThreads, 1)
void fused_router(const __nv_bfloat16* __restrict__ x,
                  const __nv_bfloat16* __restrict__ w,
                  const float* __restrict__ bias,
                  int32_t* __restrict__ ids, float* __restrict__ weights) {
    __shared__ __nv_bfloat162 sx[kDim / 2];
    __shared__ float logits[kExperts];
    __shared__ double score[kExperts];
    __shared__ double biased[kExperts];
    __shared__ double selected[kTopK];

    const int tid = threadIdx.x;
    const int token = blockIdx.x;
    const auto* xp = x + token * kDim;
    for (int i = tid; i < kDim / 2; i += kThreads) sx[i] = load_pair<Aligned>(xp, i);
    __syncthreads();

    const int group = tid / kExpertThreads;
    const int lane = tid % kExpertThreads;
    // A half warp owns an expert, with two accumulators corresponding exactly
    // to lanes 2*lane and 2*lane+1 in ops::bf16_gemv_k. Packed BF16 loads halve
    // the weight-load count without reordering either reference FMA chain.
    for (int e = group; e < kExperts; e += kExpertGroups) {
        const auto* wp = w + e * kDim;
        float even = 0.0f, odd = 0.0f;
#pragma unroll 4
        for (int j = lane; j < kDim / 2; j += kExpertThreads) {
            const float2 xv = __bfloat1622float2(sx[j]);
            const float2 wv = __bfloat1622float2(load_pair<Aligned>(wp, j));
            even = __fmaf_rn(xv.x, wv.x, even);
            odd = __fmaf_rn(xv.y, wv.y, odd);
        }
        // The reference's 16,8,4,2,1 shuffle tree becomes 8,4,2,1 on each
        // parity followed by even+odd. The FP32 logit is bitwise unchanged.
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            even += __shfl_down_sync(kWarpMask, even, off, kExpertThreads);
            odd += __shfl_down_sync(kWarpMask, odd, off, kExpertThreads);
        }
        if (lane == 0) {
            const float z = even + odd;
            logits[e] = z;
        }
    }
    __syncthreads();

    // expf can become subnormal or zero while sqrt(softplus(z)) is still a
    // substantial nonzero score. Transform in double before adding the bias;
    // never use an FP32 approximation to discard a ranking candidate.
    if (tid < kExperts) {
        const double s = reference_score(logits[tid]);
        score[tid] = s;
        biased[tid] = s + static_cast<double>(bias[tid]);
    }
    __syncthreads();

    // Selection needs only one warp. Each lane owns 12 experts; its bit mask
    // removes previous winners without changing any input or shared score.
    if (tid >= 32) return;
    unsigned removed = 0;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        int best = -1;
#pragma unroll
        for (int i = 0; i < kExperts / 32; ++i) {
            const int e = tid + i * 32;
            if (!(removed & (1u << i)) && better(e, best, biased)) best = e;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const int other = __shfl_down_sync(kWarpMask, best, off);
            if (tid + off < 32 && better(other, best, biased)) best = other;
        }
        const int winner = __shfl_sync(kWarpMask, best, 0);
        if ((winner & 31) == tid) removed |= 1u << (winner >> 5);
        if (tid == rank) {
            ids[token * kTopK + rank] = winner;
            selected[rank] = score[winner];
        }
    }
    __syncwarp(kWarpMask);
    // Match the reference's ordered double sum and division, including epsilon.
    if (tid < kTopK) {
        double sum = 0.0;
#pragma unroll
        for (int i = 0; i < kTopK; ++i) sum += selected[i];
        weights[token * kTopK + tid] = static_cast<float>(selected[tid] / (sum + 1e-20) * 1.5);
    }
}

}  // namespace

void router_topk(const __nv_bfloat16* x, int m, const __nv_bfloat16* w, const float* bias,
                 int32_t* ids, float* weights, cudaStream_t stream) {
    if (m <= 0) return;
    const bool aligned = ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) & 3u) == 0;
    if (aligned) fused_router<true><<<m, kThreads, 0, stream>>>(x, w, bias, ids, weights);
    else fused_router<false><<<m, kThreads, 0, stream>>>(x, w, bias, ids, weights);
}

}  // namespace strata::ds41::kernels
