// K15-02: exact FP32 dots, tiling 16 tokens x 3 weight rows per producer CTA.
// Caller-owned warp partials join two graph-safe launches at full-tile sizes.
#include "strata/ds41/kernels/k15_hc_prefill.hpp"
#include "strata/ds41/config.hpp"
#include "k15/exact_tile.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace strata::ds41::kernels {
namespace {
using namespace k15_detail;
constexpr unsigned kMask = 0xffffffffu;
constexpr int kFinishThreads = 1024;
static_assert(kHc == 4 && kDim == 5120 && kHcMix == kRows);
static_assert(kSinkhornIters == 20 && kWidth == kHc * kDim);
static_assert(kDotWarps % kTileWarps == 0 && kRows % kTileRows == 0);
static_assert(kDim % kFinishThreads == 0);

void fail(const char* what) {
    std::fprintf(stderr, "ds41 k15: %s\n", what);
    std::abort();
}
void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ds41 k15: %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(kMask, x, off);
    return x;
}

// Weights stay FP32, and x is converted exactly from BF16. Sharing across
// both axes reduces loads without using TF32, BF16 weights or tensor cores.
template <int Tokens, int Rows, bool Tail>
struct TileLoad {
    const __nv_bfloat16* x;
    const float* fn;
    int token_begin, row_begin, original_lane, m;
    __device__ __forceinline__ float weight(int row, int step) const {
        return fn[(row_begin + row) * kWidth + original_lane + step * kDotLanes];
    }
    __device__ __forceinline__ float value(int token, int step) const {
        if constexpr (Tail)
            if (token_begin + token >= m) return 0.0f;
        return __bfloat162float(x[static_cast<size_t>(token_begin + token) * kWidth +
                                  original_lane + step * kDotLanes]);
    }
};
struct Fma {
    __device__ __forceinline__ float operator()(float x, float w, float acc) const {
        return fmaf(x, w, acc);
    }
};

template <int Tokens, int Rows, int Warps, bool Tail>
__global__ void hc_tiled_dots(const __nv_bfloat16* __restrict__ x, int m,
                              const float* __restrict__ fn, float* __restrict__ partials,
                              int first_token) {
    const int warp = blockIdx.x * Warps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int row_begin = blockIdx.y * Rows;
    const int token_begin = first_token + blockIdx.z * Tokens;
    float dots[Rows][Tokens] = {};
    k15_detail::accumulate_tile<Tokens, Rows>(
        TileLoad<Tokens, Rows, Tail>{x, fn, token_begin, row_begin, warp * 32 + lane, m},
        Fma{}, dots);
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            // All lanes participate, including inactive tail tokens; only the
            // store is guarded. No warp primitive is reached divergently.
            const float dot = warp_sum(dots[row][token]);
            if (lane == 0 && (!Tail || token_begin + token < m))
                partials[partial_index(token_begin + token, row_begin + row, warp)] = dot;
        }
    }
}

__device__ __forceinline__ float row_sum(float x, int lane) {
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kHc; ++k) sum += __shfl_sync(kMask, x, (lane & 12) + k);
    return sum;
}
__device__ __forceinline__ float column_sum(float x, int lane) {
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kHc; ++j) sum += __shfl_sync(kMask, x, j * kHc + (lane & 3));
    return sum;
}
__device__ __forceinline__ void coefficients(const float* mixes, float r,
                                             const float* scale, const float* base,
                                             float* pre, float* post, float* comb) {
    const int lane = threadIdx.x & 31;
    if (lane < kHc) {
        const float pm = __fmul_rn(mixes[lane], r);
        const float qm = __fmul_rn(mixes[lane + kHc], r);
        pre[lane] = 1.0f / (1.0f + expf(-(pm * scale[0] + base[lane]))) + kHcEps;
        post[lane] = 2.0f / (1.0f + expf(-(qm * scale[1] + base[lane + kHc])));
    }
    const int index = lane & 15;
    const float mix = __fmul_rn(mixes[2 * kHc + index], r);
    float c = mix * scale[2] + base[2 * kHc + index];
    float maximum = -INFINITY;
#pragma unroll
    for (int k = 0; k < kHc; ++k)
        maximum = fmaxf(maximum, __shfl_sync(kMask, c, (lane & 12) + k));
    c = expf(c - maximum);
    c = c / row_sum(c, lane) + kHcEps;
    c /= column_sum(c, lane) + kHcEps;
#pragma unroll 1
    for (int i = 0; i < kSinkhornIters - 1; ++i) {
        c /= row_sum(c, lane) + kHcEps;
        c /= column_sum(c, lane) + kHcEps;
    }
    if (lane < kHc * kHc) comb[lane] = c;
}

// One CTA per token. The norm keeps the original 1024-lane, 20-term FMA
// chains and the original warp-tree/sequential-warp reduction. Each loaded x
// also feeds one of five independent hc_pre accumulators in j=0..3 order.
__global__ void hc_finish(const __nv_bfloat16* __restrict__ x,
                          const float* __restrict__ scale, const float* __restrict__ base,
                          const float* __restrict__ pre_in, __nv_bfloat16* __restrict__ y,
                          float* __restrict__ pre, float* __restrict__ post,
                          float* __restrict__ comb, const float* __restrict__ partials) {
    __shared__ float squares[32];
    __shared__ float mixes[kRows];
    __shared__ float reciprocal_rms;
    const int tid = threadIdx.x;
    const int token = blockIdx.x;
    x += static_cast<size_t>(token) * kWidth;
    y += static_cast<size_t>(token) * kDim;
    pre_in += token * kHc;
    float ss = 0.0f;
    float collapse[kDim / kFinishThreads] = {};
#pragma unroll
    for (int j = 0; j < kHc; ++j) {
        const float p = pre_in[j];
#pragma unroll
        for (int k = 0; k < kDim / kFinishThreads; ++k) {
            const float value = __bfloat162float(x[j * kDim + k * kFinishThreads + tid]);
            ss = fmaf(value, value, ss);
            collapse[k] = fmaf(p, value, collapse[k]);
        }
    }
#pragma unroll
    for (int k = 0; k < kDim / kFinishThreads; ++k)
        y[k * kFinishThreads + tid] = __float2bfloat16_rn(collapse[k]);
    ss = warp_sum(ss);
    if ((tid & 31) == 0) squares[tid >> 5] = ss;
    if (tid < kRows) {
        float dot = 0.0f;
#pragma unroll
        for (int w = 0; w < kDotWarps; ++w) dot += partials[partial_index(token, tid, w)];
        mixes[tid] = dot;
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 32; ++w) sum += squares[w];
        reciprocal_rms = rsqrtf(sum / static_cast<float>(kWidth) + kNormEps);
    }
    __syncthreads();
    if (tid < 32) coefficients(mixes, reciprocal_rms, scale, base,
                              pre + token * kHc, post + token * kHc, comb + token * kHc * kHc);
}
}  // namespace

size_t hc_mixes_pre_rows_workspace_bytes(int m) { return k15_detail::workspace_size(m); }

void hc_mixes_pre_rows(const __nv_bfloat16* x, int m, const float* fn, const float* scale, const float* base,
                       const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb, void* workspace,
                       size_t workspace_bytes, cudaStream_t stream) {
    const size_t required = hc_mixes_pre_rows_workspace_bytes(m);
    if (!required) fail("invalid token count (expected 1..16384)");
    if (!workspace || workspace_bytes < required ||
        (reinterpret_cast<uintptr_t>(workspace) % alignof(float)) != 0)
        fail("workspace is null, too small or not float-aligned");
    auto* partials = static_cast<float*>(workspace);
    if (m == 1) {
        hc_tiled_dots<1, 1, 1, false><<<dim3(kDotWarps, kRows), 32, 0, stream>>>(x, m, fn, partials, 0);
        check_launch("single-token dots");
    } else {
        const int full = m / kTileTokens;
        const dim3 grid(kDotWarps / kTileWarps, kRows / kTileRows, full);
        if (full) {
            hc_tiled_dots<kTileTokens, kTileRows, kTileWarps, false>
                <<<grid, kTileWarps * 32, 0, stream>>>(x, m, fn, partials, 0);
            check_launch("tiled dots");
        }
        if (m % kTileTokens) {
            hc_tiled_dots<kTileTokens, kTileRows, kTileWarps, true>
                <<<dim3(grid.x, grid.y), kTileWarps * 32, 0, stream>>>(
                    x, m, fn, partials, full * kTileTokens);
            check_launch("tail dots");
        }
    }
    hc_finish<<<m, kFinishThreads, 0, stream>>>(x, scale, base, pre_in, y, pre, post, comb, partials);
    check_launch("norm, collapse and coefficients");
}
}  // namespace strata::ds41::kernels
