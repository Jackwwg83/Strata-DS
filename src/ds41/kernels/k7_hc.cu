// src/ds41/kernels/k7_hc.cu - exact FP32 chains with aligned four-lane packed loads.
#include "strata/ds41/kernels/k7_hc.hpp"

#include "strata/ds41/config.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::ds41::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kProducerThreads = 32;
constexpr int kDotThreads = 256;
constexpr int kDotWarps = kDotThreads / 32;
constexpr int kNormThreads = 1024;
constexpr int kNormWarps = kNormThreads / 32;
constexpr int kStreamSize = kHc * kDim;
constexpr int kReductionRows = kHcMix + 1;
constexpr int kMaxTokens = 8;  // The fixed interface permits m in [1, 8].
constexpr unsigned kWarpMask = 0xffffffffu;
static_assert(kHc == 4 && kHcMix == 24);
static_assert(kDim % kThreads == 0 && kStreamSize % kNormThreads == 0);
static_assert(kNormThreads == kHc * kDotThreads);

struct Workspace {
    float dots[kMaxTokens][kHcMix][kDotWarps];
    float squares[kMaxTokens][kNormWarps];
};
static_assert(sizeof(Workspace) == 7168);

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "ds41 k7: %s: %s\n", operation, cudaGetErrorString(status));
        std::abort();
    }
}

// K7 owns one maximum-size allocation PER DEVICE, retained for the process.
// The engine guarantees nonoverlapping K7 calls/replays on each device, even
// across streams. Other task kernels have their own, unrelated workspaces.
// The required eager call initializes this allocation before graph capture;
// subsequent calls only look it up, including when m or the stream changes.
Workspace* workspace_for_device() {
    struct Entry {
        int device;
        Workspace* workspace;
    };
    static std::mutex mutex;
    static std::vector<Entry> entries;
    int device = 0;
    check_cuda(cudaGetDevice(&device), "get device");
    const std::lock_guard<std::mutex> lock(mutex);
    for (const auto& entry : entries)
        if (entry.device == device) return entry.workspace;
    Workspace* workspace = nullptr;
    check_cuda(cudaMalloc(&workspace, sizeof(Workspace)), "allocate workspace");
    entries.push_back({device, workspace});
    return workspace;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(kWarpMask, value, offset);
    return value;
}

// Each warp-only CTA owns one ORIGINAL reference warp of one weight row.
// Each lane retains the complete 80-term stride-256 FMA chain. Splitting a
// chain into contiguous tiles changes FP32 cancellation and is not equivalent.
// Every weight is loaded once and reused for all tokens. No shared-memory
// reduction is needed: the original shuffle tree produces each warp total.
// The first four rows also own the original 1024 norm lanes, partitioned by
// step mod 4, preserving their stride-1024 chains and 32 reference warp totals.
template <int Tokens>
__global__ void hc_partials(const __nv_bfloat16* __restrict__ x,
                            const float* __restrict__ fn, Workspace* workspace) {
    const int lane = threadIdx.x;
    const int warp = blockIdx.x;
    const int row = blockIdx.y;
    const int original_lane = warp * 32 + lane;
    float dots[Tokens] = {};
    float squares[Tokens] = {};
#pragma unroll 1
    for (int step = 0; step < kStreamSize / kDotThreads; ++step) {
        const int column = original_lane + step * kDotThreads;
        const float weight = fn[row * kStreamSize + column];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const float value = __bfloat162float(x[token * kStreamSize + column]);
            dots[token] = fmaf(value, weight, dots[token]);
            if (row < kHc && (step & (kHc - 1)) == row)
                squares[token] = fmaf(value, value, squares[token]);
        }
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float dot = warp_sum(dots[token]);
        if (lane == 0) workspace->dots[token][row][warp] = dot;
        if (row < kHc) {
            const float square = warp_sum(squares[token]);
            if (lane == 0)
                workspace->squares[token][row * kDotWarps + warp] = square;
        }
    }
}

// One physical lane holds four ADJACENT original lanes. The original offsets
// 16, 8, 4 become offsets 4, 2, 1 among eight physical lanes, independently in
// each component. Only afterward do original offsets 2 and 1 combine components.
// Summing the four components first, or summing x+y+z+w, changes cancellation.
// Every physical lane executes every shuffle; width 8 isolates original warps.
__device__ __forceinline__ float packed_warp_sum(float4 v) {
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        v.x = __fadd_rn(v.x, __shfl_down_sync(kWarpMask, v.x, offset, 8));
        v.y = __fadd_rn(v.y, __shfl_down_sync(kWarpMask, v.y, offset, 8));
        v.z = __fadd_rn(v.z, __shfl_down_sync(kWarpMask, v.z, offset, 8));
        v.w = __fadd_rn(v.w, __shfl_down_sync(kWarpMask, v.w, offset, 8));
    }
    return __fadd_rn(__fadd_rn(v.x, v.z), __fadd_rn(v.y, v.w));
}

__device__ __forceinline__ float4 load_bf16x4(const __nv_bfloat16* values) {
    const uint2 packed = __ldg(reinterpret_cast<const uint2*>(values));
    return make_float4(
        __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed.x))),
        __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed.x >> 16))),
        __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed.y))),
        __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed.y >> 16))));
}

// Four original warps per CTA, eight physical lanes per original warp. The
// 80-step stride-256 FMA chain and RMS step-mod-four ownership are unchanged.
// One float4 weight and one uint2 activation per token replace four scalar
// loads each. All Tokens reuse every weight component. This is a load-issue
// ablation: 48 CTAs instead of 192, with four times the accumulator state per
// thread. It trades away thread-level parallelism; GPU timing must decide.
template <int Tokens>
__global__ void hc_packed_partials(const __nv_bfloat16* __restrict__ x,
                                  const float* __restrict__ fn, Workspace* workspace) {
    const int lane = threadIdx.x;
    const int warp = blockIdx.x * 4 + lane / 8;
    const int row = blockIdx.y;
    const int original_lane = warp * 32 + (lane & 7) * 4;
    float4 dots[Tokens] = {};
    float4 squares[Tokens] = {};
#pragma unroll 1
    for (int step = 0; step < kStreamSize / kDotThreads; ++step) {
        const int column = original_lane + step * kDotThreads;
        const float4 weight = __ldg(reinterpret_cast<const float4*>(
            fn + row * kStreamSize + column));
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const float4 value = load_bf16x4(x + token * kStreamSize + column);
            dots[token].x = fmaf(value.x, weight.x, dots[token].x);
            dots[token].y = fmaf(value.y, weight.y, dots[token].y);
            dots[token].z = fmaf(value.z, weight.z, dots[token].z);
            dots[token].w = fmaf(value.w, weight.w, dots[token].w);
            if (row < kHc && (step & (kHc - 1)) == row) {
                squares[token].x = fmaf(value.x, value.x, squares[token].x);
                squares[token].y = fmaf(value.y, value.y, squares[token].y);
                squares[token].z = fmaf(value.z, value.z, squares[token].z);
                squares[token].w = fmaf(value.w, value.w, squares[token].w);
            }
        }
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float dot = packed_warp_sum(dots[token]);
        if ((lane & 7) == 0) workspace->dots[token][row][warp] = dot;
        if (row < kHc) {
            const float square = packed_warp_sum(squares[token]);
            if ((lane & 7) == 0)
                workspace->squares[token][row * kDotWarps + warp] = square;
        }
    }
}

// All 32 lanes execute each shuffle. Lanes 0..15 own the 4x4 matrix;
// lanes 16..31 duplicate it to keep the warp converged. Sum four entries in
// reference order, rather than changing the parenthesization to a tree sum.
__device__ __forceinline__ float row_sum(float value, int lane) {
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kHc; ++k)
        sum += __shfl_sync(kWarpMask, value, (lane & 12) + k);
    return sum;
}

__device__ __forceinline__ float column_sum(float value, int lane) {
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kHc; ++j)
        sum += __shfl_sync(kWarpMask, value, j * kHc + (lane & 3));
    return sum;
}

__device__ __forceinline__ void finish_coefficients(
    const float* mixes, float reciprocal_rms, const float* scale, const float* base,
    float* pre, float* post, float* comb) {
    const int lane = threadIdx.x & 31;
    if (lane < kHc) {
        // Preserve the FP32 rounding between normalization and scale/bias.
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
        maximum = fmaxf(maximum, __shfl_sync(kWarpMask, c, (lane & 12) + k));
    c = expf(c - maximum);
    c = c / row_sum(c, lane) + kHcEps;

    // Exactly the reference: initial column normalization, followed by 19
    // row/column pairs. The epsilon stays in EVERY normalization denominator.
    c = c / (column_sum(c, lane) + kHcEps);
#pragma unroll 1
    for (int iteration = 0; iteration < kSinkhornIters - 1; ++iteration) {
        c = c / (row_sum(c, lane) + kHcEps);
        c = c / (column_sum(c, lane) + kHcEps);
    }
    if (lane < kHc * kHc) comb[lane] = c;
}

// Launch ordering on the supplied stream publishes ALL partials before this
// stage. There is no cross-CTA spin, fence, atomic, or in-kernel global barrier.
// The feature tiles also distribute collapse across 20 CTAs per token.
__global__ void hc_finish(const __nv_bfloat16* __restrict__ x,
                          const float* __restrict__ scale, const float* __restrict__ base,
                          const float* __restrict__ pre_in, __nv_bfloat16* __restrict__ y,
                          float* __restrict__ pre, float* __restrict__ post,
                          float* __restrict__ comb, const Workspace* workspace) {
    __shared__ float mixes[kHcMix];
    __shared__ float reciprocal_rms;
    const int tid = threadIdx.x;
    const int token = blockIdx.y;
    const int d = blockIdx.x * kThreads + tid;

    // Match ops::hc_pre's j=0..3 FP32 accumulation and single BF16 rounding.
    float collapsed = 0.0f;
#pragma unroll
    for (int j = 0; j < kHc; ++j)
        collapsed += pre_in[token * kHc + j] *
                     __bfloat162float(x[token * kStreamSize + j * kDim + d]);
    y[token * kDim + d] = __float2bfloat16_rn(collapsed);
    if (blockIdx.x != 0) return;  // Uniform for the CTA, before any barrier.

    if (tid < kReductionRows) {
        // Start at +0 and sum warp totals in exactly ops::block_sum order.
        float sum = 0.0f;
        if (tid == kHcMix) {
#pragma unroll
            for (int w = 0; w < kNormWarps; ++w)
                sum += workspace->squares[token][w];
            reciprocal_rms = rsqrtf(sum / static_cast<float>(kStreamSize) + kNormEps);
        } else {
#pragma unroll
            for (int w = 0; w < kDotWarps; ++w)
                sum += workspace->dots[token][tid][w];
            mixes[tid] = sum;
        }
    }
    __syncthreads();
    if (tid < 32)
        finish_coefficients(mixes, reciprocal_rms, scale, base, pre + token * kHc,
                            post + token * kHc, comb + token * kHc * kHc);
}

}  // namespace

void hc_mixes_pre(const __nv_bfloat16* x, int m, const float* fn, const float* scale, const float* base,
                  const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb,
                  cudaStream_t stream) {
    if (m < 1 || m > kMaxTokens) {
        std::fprintf(stderr, "ds41 k7: invalid token count %d (expected 1..8)\n", m);
        std::abort();
    }
    Workspace* workspace = workspace_for_device();
    // Only the aligned path widens memory accesses. All row/token/step strides
    // preserve these alignments; naturally aligned BF16/float offset views use
    // the original scalar producer with no extra copy or allocation.
    const bool packed = (reinterpret_cast<std::uintptr_t>(fn) % alignof(float4) == 0) &&
                        (reinterpret_cast<std::uintptr_t>(x) % alignof(uint2) == 0);
    const dim3 partial_grid(packed ? kDotWarps / 4 : kDotWarps, kHcMix);
#define K7_LAUNCH(TOKENS) \
    case TOKENS: \
        if (packed) \
            hc_packed_partials<TOKENS><<<partial_grid, kProducerThreads, 0, stream>>>(x, fn, workspace); \
        else \
            hc_partials<TOKENS><<<partial_grid, kProducerThreads, 0, stream>>>(x, fn, workspace); \
        break
    switch (m) {
        K7_LAUNCH(1);
        K7_LAUNCH(2);
        K7_LAUNCH(3);
        K7_LAUNCH(4);
        K7_LAUNCH(5);
        K7_LAUNCH(6);
        K7_LAUNCH(7);
        K7_LAUNCH(8);
    }
#undef K7_LAUNCH
    check_cuda(cudaGetLastError(), "launch partial mixes");
    hc_finish<<<dim3(kDim / kThreads, m), kThreads, 0, stream>>>(
        x, scale, base, pre_in, y, pre, post, comb, workspace);
    check_cuda(cudaGetLastError(), "launch reduction and collapse");
}

}  // namespace strata::ds41::kernels
