// One CTA per token: cache the stream, compute all 24 FP32 mixes, and finish
// Sinkhorn and collapse without global scratch or host-side initialization.
#include "strata/ds41/kernels/k7_hc.hpp"

#include "strata/ds41/config.hpp"

namespace strata::ds41::kernels {
namespace {

constexpr int kHcd = kHc * kDim;
constexpr int kThreads = 1024;
constexpr int kWarp = 32;
constexpr int kDotWarps = 256 / kWarp;
static_assert(kHc == 4 && kHcMix == 24, "This kernel implements the fixed K7 geometry");
static_assert(kHcd % kThreads == 0 && kHcd % 256 == 0, "Complete RMS/dot stripes required");

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// Keep the reference's scalar operation order, including the additive epsilons:
// row softmax + epsilon, column normalization, then 19 row/column pairs.
__device__ __forceinline__ void sinkhorn(const float* mix, const float* scale, const float* base,
                                        float* pre, float* post, float* comb) {
    for (int j = 0; j < kHc; ++j) {
        pre[j] = 1.0f / (1.0f + expf(-(mix[j] * scale[0] + base[j]))) + kHcEps;
        post[j] = 2.0f / (1.0f + expf(-(mix[j + kHc] * scale[1] + base[j + kHc])));
    }
    float c[kHc][kHc];
    for (int j = 0; j < kHc; ++j) {
        float mx = -INFINITY;
        for (int k = 0; k < kHc; ++k) {
            c[j][k] = mix[2 * kHc + j * kHc + k] * scale[2] + base[2 * kHc + j * kHc + k];
            mx = fmaxf(mx, c[j][k]);
        }
        float s = 0.0f;
        for (int k = 0; k < kHc; ++k) { c[j][k] = expf(c[j][k] - mx); s += c[j][k]; }
        for (int k = 0; k < kHc; ++k) c[j][k] = c[j][k] / s + kHcEps;
    }
    auto col_norm = [&]() {
        for (int k = 0; k < kHc; ++k) {
            float s = 0.0f;
            for (int j = 0; j < kHc; ++j) s += c[j][k];
            for (int j = 0; j < kHc; ++j) c[j][k] = c[j][k] / (s + kHcEps);
        }
    };
    col_norm();
    for (int it = 0; it < kSinkhornIters - 1; ++it) {
        for (int j = 0; j < kHc; ++j) {
            float s = 0.0f;
            for (int k = 0; k < kHc; ++k) s += c[j][k];
            for (int k = 0; k < kHc; ++k) c[j][k] = c[j][k] / (s + kHcEps);
        }
        col_norm();
    }
    for (int j = 0; j < kHc; ++j)
        for (int k = 0; k < kHc; ++k) comb[j * kHc + k] = c[j][k];
}

__global__ __launch_bounds__(kThreads, 1) void hc_fused_token(
    const __nv_bfloat16* x, const float* fn, const float* scale, const float* base,
    const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb) {
    // Preserve BF16 exactly and stay below the 48-KiB default shared-memory limit;
    // no cudaFuncSetAttribute/device-specific first-call state is necessary.
    __shared__ __nv_bfloat16 cached[kHcd];
    __shared__ float warp_sums[kThreads / kWarp];
    __shared__ float mixes[kHcMix];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int token = blockIdx.x;
    x += static_cast<size_t>(token) * kHcd;
    y += static_cast<size_t>(token) * kDim;
    pre_in += token * kHc;

    // Exactly the reference's 1024-thread RMS partition and reduction tree.
    float ss = 0.0f;
    for (int i = tid; i < kHcd; i += kThreads) {
        const __nv_bfloat16 v = x[i];
        cached[i] = v;
        const float f = __bfloat162float(v);
        ss = __fmaf_rn(f, f, ss);
    }
    ss = warp_sum(ss);
    if (lane == 0) warp_sums[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < kThreads / kWarp; ++w) total += warp_sums[w];
        warp_sums[0] = rsqrtf(total / static_cast<float>(kHcd) + kNormEps);
    }
    __syncthreads();

    if (warp < kHcMix) {
        // One physical warp owns one output row. Each lane holds the eight
        // virtual-warp accumulators from the reference's 256-thread dot CTA.
        // This gives all 24 rows concurrent memory requests and eight independent
        // FP32 FMA chains per lane without reassociating the reference dot sum.
        const float* row = fn + static_cast<size_t>(warp) * kHcd;
        float acc[kDotWarps] = {};
#pragma unroll 1
        for (int i = lane; i < kHcd; i += 256) {
#pragma unroll
            for (int w = 0; w < kDotWarps; ++w) {
                const int col = i + w * kWarp;
                acc[w] = __fmaf_rn(__bfloat162float(cached[col]), row[col], acc[w]);
            }
        }
#pragma unroll
        for (int w = 0; w < kDotWarps; ++w) acc[w] = warp_sum(acc[w]);
        if (lane == 0) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < kDotWarps; ++w) sum += acc[w];
            mixes[warp] = sum * warp_sums[0];
        }
    } else {
        // The remaining eight warps perform the independent collapse while the
        // first 24 read weights. The original j order and BF16 rounding remain.
        constexpr int kCollapseThreads = kThreads - kHcMix * kWarp;
        for (int d = tid - kHcMix * kWarp; d < kDim; d += kCollapseThreads) {
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < kHc; ++j)
                sum = __fmaf_rn(pre_in[j], __bfloat162float(cached[j * kDim + d]), sum);
            y[d] = __float2bfloat16_rn(sum);
        }
    }
    __syncthreads();
    if (tid == 0)
        sinkhorn(mixes, scale, base, pre + token * kHc, post + token * kHc,
                 comb + token * kHc * kHc);
}

}  // namespace

void hc_mixes_pre(const __nv_bfloat16* x, int m, const float* fn, const float* scale, const float* base,
                  const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb,
                  cudaStream_t stream) {
    if (m <= 0) return;
    // m is 1..8 by contract. Only m CTAs is an intentional latency/occupancy
    // tradeoff: this path removes every intermediate launch and allocation, but
    // the small grid may underutilize the GPU. Timing needs the GPU test queue.
    hc_fused_token<<<m, kThreads, 0, stream>>>(x, fn, scale, base, pre_in, y, pre, post, comb);
}

}  // namespace strata::ds41::kernels
