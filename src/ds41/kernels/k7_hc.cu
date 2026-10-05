// src/ds41/kernels/k7_hc.cu - batched FP32 mixes with a stream-ordered reduction.
#include "strata/ds41/kernels/k7_hc.hpp"

#include "strata/ds41/config.hpp"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::ds41::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunk = 1024;
constexpr int kStreamSize = kHc * kDim;
constexpr int kParts = kStreamSize / kChunk;
constexpr int kReductionRows = kHcMix + 1;
constexpr int kMaxTokens = 8;  // The fixed interface permits m in [1, 8].
constexpr unsigned kWarpMask = 0xffffffffu;
static_assert(kHc == 4 && kHcMix == 24);
static_assert(kStreamSize % kChunk == 0 && kDim % kThreads == 0);

struct Workspace {
    float partial[kMaxTokens][kReductionRows][kParts];
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

// Each CTA owns a disjoint 1024-column tile of one FP32 weight row. Every
// weight is loaded exactly once, then reused in registers for ALL m tokens.
// Splitting the long dot products exposes 480 independent CTAs even for m=1.
// Row zero also computes the norm; no float atomics or completion counters.
template <int Tokens>
__global__ void hc_partials(const __nv_bfloat16* __restrict__ x,
                            const float* __restrict__ fn, Workspace* workspace) {
    __shared__ float warp_dots[Tokens][kWarps];
    __shared__ float warp_squares[Tokens][kWarps];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.y;
    const int part = blockIdx.x;
    float dots[Tokens] = {};
    float squares[Tokens] = {};
#pragma unroll
    for (int offset = 0; offset < kChunk; offset += kThreads) {
        const int column = part * kChunk + offset + tid;
        const float weight = fn[row * kStreamSize + column];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const float value = __bfloat162float(x[token * kStreamSize + column]);
            dots[token] += value * weight;
            if (row == 0) squares[token] += value * value;
        }
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float dot = warp_sum(dots[token]);
        if (lane == 0) warp_dots[token][warp] = dot;
        if (row == 0) {
            const float square = warp_sum(squares[token]);
            if (lane == 0) warp_squares[token][warp] = square;
        }
    }
    __syncthreads();

    // One thread per token sums the eight warp totals in a fixed order.
    if (tid < Tokens) {
        float dot = 0.0f;
        float square = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            dot += warp_dots[tid][w];
            if (row == 0) square += warp_squares[tid][w];
        }
        workspace->partial[tid][row][part] = dot;
        if (row == 0) workspace->partial[tid][kHcMix][part] = square;
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
        float sum = 0.0f;
#pragma unroll
        for (int part = 0; part < kParts; ++part)
            sum += workspace->partial[token][tid][part];
        if (tid == kHcMix)
            reciprocal_rms = rsqrtf(sum / static_cast<float>(kStreamSize) + kNormEps);
        else
            mixes[tid] = sum;
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
    const dim3 partial_grid(kParts, kHcMix);
#define K7_LAUNCH(TOKENS) \
    case TOKENS: \
        hc_partials<TOKENS><<<partial_grid, kThreads, 0, stream>>>(x, fn, workspace); \
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
