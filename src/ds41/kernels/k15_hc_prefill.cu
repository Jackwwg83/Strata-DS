// K15-01: one FP32 SIMT CTA per token, with exact reference reduction order.
// All tokens share a single launch; dot/norm finalization, Sinkhorn and hc_pre
// are fused. No cross-CTA state or workspace is needed.
#include "strata/ds41/kernels/k15_hc_prefill.hpp"

#include "strata/ds41/config.hpp"
#include "k15/exact_accumulate.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::ds41::kernels {
namespace {
constexpr int kThreads = 256;
constexpr int kDotWarps = kThreads / 32;
constexpr int kNormWarps = 1024 / 32;
constexpr int kStreamSize = kHc * kDim;
constexpr unsigned kMask = 0xffffffffu;
static_assert(kHc == 4 && kHcMix == 24 && kDim == 5120);
static_assert(kStreamSize == k15_detail::kColumns && kHcMix == k15_detail::kRows);

struct Load {
    const __nv_bfloat16* x;
    const float* fn;
    int tid;
    __device__ __forceinline__ float value(int step) const {
        return __bfloat162float(x[tid + step * kThreads]);
    }
    __device__ __forceinline__ float weight(int row, int step) const {
        return fn[row * kStreamSize + tid + step * kThreads];
    }
};
struct Fma {
    __device__ __forceinline__ float operator()(float a, float b, float c) const {
        return fmaf(a, b, c);
    }
};
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(kMask, value, offset);
    return value;
}

// Every lane participates. Lanes 16..31 duplicate lanes 0..15 so each shuffle
// uses a fully active mask. Four-term sums retain the reference's serial order.
__device__ __forceinline__ float row_sum(float value, int lane) {
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kHc; ++k)
        sum += __shfl_sync(kMask, value, (lane & 12) + k);
    return sum;
}
__device__ __forceinline__ float column_sum(float value, int lane) {
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kHc; ++j)
        sum += __shfl_sync(kMask, value, j * kHc + (lane & 3));
    return sum;
}
__device__ __forceinline__ void finish_coefficients(
    const float* mixes, float reciprocal_rms, const float* scale, const float* base,
    float* pre, float* post, float* comb) {
    const int lane = threadIdx.x & 31;
    if (lane < kHc) {
        // ops::hc_dot_k stores the normalized mix in FP32 before scale/bias.
        // Explicit multiplication rounding prevents reassociation across it.
        const float pm = __fmul_rn(mixes[lane], reciprocal_rms);
        const float qm = __fmul_rn(mixes[lane + kHc], reciprocal_rms);
        pre[lane] = 1.0f / (1.0f + expf(-(pm * scale[0] + base[lane]))) + kHcEps;
        post[lane] = 2.0f / (1.0f + expf(-(qm * scale[1] + base[lane + kHc])));
    }
    const int index = lane & 15;
    const float mix = __fmul_rn(mixes[2 * kHc + index], reciprocal_rms);
    float c = mix * scale[2] + base[2 * kHc + index];
    float maximum = -INFINITY;
#pragma unroll
    for (int k = 0; k < kHc; ++k)
        maximum = fmaxf(maximum, __shfl_sync(kMask, c, (lane & 12) + k));
    c = expf(c - maximum);
    c = c / row_sum(c, lane) + kHcEps;
    c = c / (column_sum(c, lane) + kHcEps);
#pragma unroll 1
    for (int iteration = 0; iteration < kSinkhornIters - 1; ++iteration) {
        c = c / (row_sum(c, lane) + kHcEps);
        c = c / (column_sum(c, lane) + kHcEps);
    }
    if (lane < kHc * kHc) comb[lane] = c;
}

__global__ void hc_rows(const __nv_bfloat16* __restrict__ x,
                        const float* __restrict__ fn, const float* __restrict__ scale,
                        const float* __restrict__ base, const float* __restrict__ pre_in,
                        __nv_bfloat16* __restrict__ y, float* __restrict__ pre,
                        float* __restrict__ post, float* __restrict__ comb) {
    __shared__ float partials[kHcMix][kDotWarps];
    __shared__ float norm_partials[kNormWarps];
    __shared__ float mixes[kHcMix];
    __shared__ float reciprocal_rms;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int token = blockIdx.x;
    x += static_cast<size_t>(token) * kStreamSize;
    pre_in += token * kHc;
    y += static_cast<size_t>(token) * kDim;

    float dots[kHcMix] = {};
    float squares[kHc] = {};
    k15_detail::accumulate(Load{x, fn, tid}, Fma{}, dots, squares);
#pragma unroll
    for (int row = 0; row < kHcMix; ++row) {
        const float dot = warp_sum(dots[row]);
        if (lane == 0) partials[row][warp] = dot;
    }
#pragma unroll
    for (int phase = 0; phase < kHc; ++phase) {
        const float square = warp_sum(squares[phase]);
        if (lane == 0) norm_partials[phase * kDotWarps + warp] = square;
    }
    __syncthreads();

    // Serial sums start at +0 and consume reference warp totals in order.
    if (tid < kHcMix) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kDotWarps; ++w) sum += partials[tid][w];
        mixes[tid] = sum;
    } else if (tid == kHcMix) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kNormWarps; ++w) sum += norm_partials[w];
        reciprocal_rms = rsqrtf(sum / static_cast<float>(kStreamSize) + kNormEps);
    }
    __syncthreads();

    // Independent of the new coefficients: collapse uses the incoming pre.
    // Keep j=0..3 FP32 FMA order, followed by exactly one BF16 rounding.
    for (int d = tid; d < kDim; d += kThreads) {
        float collapsed = 0.0f;
#pragma unroll
        for (int j = 0; j < kHc; ++j)
            collapsed = fmaf(pre_in[j], __bfloat162float(x[j * kDim + d]), collapsed);
        y[d] = __float2bfloat16_rn(collapsed);
    }
    if (tid < 32)
        finish_coefficients(mixes, reciprocal_rms, scale, base, pre + token * kHc,
                            post + token * kHc, comb + token * kHc * kHc);
}
}  // namespace

size_t hc_mixes_pre_rows_workspace_bytes(int m) {
    (void) m;
    return 0;
}
void hc_mixes_pre_rows(const __nv_bfloat16* x, int m, const float* fn, const float* scale, const float* base,
                       const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb, void* workspace,
                       size_t workspace_bytes, cudaStream_t stream) {
    (void) workspace, (void) workspace_bytes;
    if (m < 1 || m > 16384) {
        std::fprintf(stderr, "ds41 k15: invalid token count %d (expected 1..16384)\n", m);
        std::abort();
    }
    hc_rows<<<m, kThreads, 0, stream>>>(x, fn, scale, base, pre_in, y, pre, post, comb);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "ds41 k15: launch: %s\n", cudaGetErrorString(error));
        std::abort();
    }
}
}  // namespace strata::ds41::kernels
