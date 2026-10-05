// src/ds41/kernels/k7_hc.cu - reference-order dots with vectorized tile reuse.
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
constexpr int kRowsPerBlock = 2;
constexpr int kPartialThreads = 32 * kRowsPerBlock;
constexpr int kDotWarps = 8;     // The reference dot kernel has 256 threads.
constexpr int kNormWarps = 32;   // The reference RMS kernel has 1024 threads.
constexpr int kTileSteps = 16;
constexpr int kTileValues = kTileSteps * 32;
constexpr int kStreamSize = kHc * kDim;
constexpr int kDotSteps = kStreamSize / 256;
constexpr int kMaxTokens = 8;  // The fixed interface permits m in [1, 8].
constexpr unsigned kWarpMask = 0xffffffffu;
static_assert(kHc == 4 && kHcMix == 24);
static_assert(kStreamSize % 1024 == 0 && kDim % kThreads == 0);
static_assert(kHcMix % kRowsPerBlock == 0 && kDotSteps % kTileSteps == 0);
static_assert(kTileSteps % 4 == 0);

struct Workspace {
    float dot[kMaxTokens][kHcMix][kDotWarps];
    float norm[kMaxTokens][kNormWarps];
};

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

// Each CTA owns two weight rows and ONE original 32-lane dot warp. The
// 12*8 grid gives 96 CTAs. Shared input tiles are reused by both row warps;
// float4 fn loads are staged into shared memory and then consumed by the
// original lanes. This changes data movement, not the FP32 reduction tree.
// Scalar loading also supports float-aligned fn subviews; x needs only BF16
// alignment. The largest shared allocation is 20 KiB (m=8).
__device__ __forceinline__ float4 load_weights(const float* pointer, bool aligned) {
    if (aligned) return *reinterpret_cast<const float4*>(pointer);
    return make_float4(pointer[0], pointer[1], pointer[2], pointer[3]);
}

template <int Tokens>
__global__ void hc_partials(const __nv_bfloat16* __restrict__ x,
                            const float* __restrict__ fn, Workspace* workspace) {
    __shared__ __align__(16) float cached_x[Tokens][kTileValues];
    __shared__ __align__(16) float cached_fn[kRowsPerBlock][kTileValues];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int row_warp = tid >> 5;
    const int row_group = blockIdx.y;
    const int row = row_group * kRowsPerBlock + row_warp;
    const int reference_warp = blockIdx.x;
    const bool aligned = (reinterpret_cast<std::uintptr_t>(fn) & 15u) == 0;
    float dots[Tokens] = {};
    float squares[2][Tokens] = {};

    // These accumulators survive EVERY tile. Each dot lane sees exactly
    // reference_warp*32 + lane + 256*step, step=0..79, in that order.
    // Splitting the 80-term chain into separately rounded partial sums is
    // invalid: large finite cancellation can exceed coefficient tolerance.
#pragma unroll 1
    for (int first_step = 0; first_step < kDotSteps; first_step += kTileSteps) {
#pragma unroll
        for (int local = tid; local < kTileValues; local += kPartialThreads) {
            const int column = reference_warp * 32 +
                               (first_step + local / 32) * 256 + (local & 31);
#pragma unroll
            for (int token = 0; token < Tokens; ++token)
                cached_x[token][local] = __bfloat162float(x[token * kStreamSize + column]);
        }
#pragma unroll
        for (int local = lane * 4; local < kTileValues; local += 32 * 4) {
            const int column = reference_warp * 32 +
                               (first_step + local / 32) * 256 + (local & 31);
            *reinterpret_cast<float4*>(&cached_fn[row_warp][local]) =
                load_weights(fn + row * kStreamSize + column, aligned);
        }
        __syncthreads();

#pragma unroll
        for (int step = 0; step < kTileSteps; ++step) {
            const int local = step * 32 + lane;
            const float weight = cached_fn[row_warp][local];
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                const float value = cached_x[token][local];
                dots[token] = __fmaf_rn(value, weight, dots[token]);
                // Only row group zero owns RMS. Its two row warps each own
                // two ORIGINAL norm warps. q=row_warp or row_warp+2 gives
                // column=(q*8+reference_warp)*32+lane+1024*iteration.
                // Each norm chain retains the reference's 20 FMA terms.
                if (row_group == 0) {
                    if ((step & 3) == row_warp)
                        squares[0][token] = __fmaf_rn(value, value, squares[0][token]);
                    if ((step & 3) == row_warp + 2)
                        squares[1][token] = __fmaf_rn(value, value, squares[1][token]);
                }
            }
        }
        // All row warps must finish reading a tile before it is overwritten.
        if (first_step + kTileSteps < kDotSteps) __syncthreads();
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float dot = warp_sum(dots[token]);
        if (lane == 0) workspace->dot[token][row][reference_warp] = dot;
        if (row_group == 0) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const float square = warp_sum(squares[q][token]);
                if (lane == 0)
                    workspace->norm[token][(row_warp + 2 * q) * 8 + reference_warp] = square;
            }
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

    if (tid < kHcMix) {
        float sum = 0.0f;
#pragma unroll
        for (int warp = 0; warp < kDotWarps; ++warp)
            sum += workspace->dot[token][tid][warp];
        mixes[tid] = sum;
    }
    if (tid == kHcMix) {
        float sum = 0.0f;
#pragma unroll
        for (int warp = 0; warp < kNormWarps; ++warp)
            sum += workspace->norm[token][warp];
        reciprocal_rms = rsqrtf(sum / static_cast<float>(kStreamSize) + kNormEps);
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
    const dim3 partial_grid(kDotWarps, kHcMix / kRowsPerBlock);
#define K7_LAUNCH(TOKENS) \
    case TOKENS: \
        hc_partials<TOKENS><<<partial_grid, kPartialThreads, 0, stream>>>(x, fn, workspace); \
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
