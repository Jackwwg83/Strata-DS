// src/ds41/kernels/k7_hc.cu - single-launch, last-completing-CTA K7 reduction.
// Ordering and lifecycle proof: k7/LAST_CTA.md.
#include "strata/ds41/kernels/k7_hc.hpp"

#include "strata/ds41/config.hpp"

#include <cuda/atomic>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::ds41::kernels {
namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kDotThreads = 256;
constexpr int kDotWarps = kDotThreads / 32;
constexpr int kNormThreads = 1024;
constexpr int kNormWarps = kNormThreads / 32;
constexpr int kStreamSize = kHc * kDim;
constexpr int kParts = kDotThreads / kThreads;
constexpr int kReductionRows = kHcMix + 1;
constexpr int kMaxTokens = 8;  // The fixed interface permits m in [1, 8].
constexpr unsigned kWarpMask = 0xffffffffu;
static_assert(kHc == 4 && kHcMix == 24);
static_assert(kDotThreads % kThreads == 0 && kDim % kThreads == 0);
static_assert(kNormThreads == kHc * kDotThreads);

using Completion = cuda::atomic_ref<unsigned, cuda::thread_scope_device>;
struct Workspace {
    float dots[kMaxTokens][kHcMix][kDotWarps];
    float squares[kMaxTokens][kNormWarps];
    alignas(Completion::required_alignment) unsigned completed;
};
static_assert(offsetof(Workspace, completed) % Completion::required_alignment == 0);
static_assert(sizeof(Workspace) == 7172);

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
Workspace* workspace_for_device(cudaStream_t stream) {
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
    // Initialization must occur in the required eager call, never in capture.
    cudaStreamCaptureStatus capture_status;
    check_cuda(cudaStreamIsCapturing(stream, &capture_status), "query initial capture");
    if (capture_status != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "ds41 k7: one eager call is required before capture\n");
        std::abort();
    }
    Workspace* workspace = nullptr;
    check_cuda(cudaMalloc(&workspace, sizeof(Workspace)), "allocate workspace");
    check_cuda(cudaMemsetAsync(&workspace->completed, 0, sizeof(unsigned), stream),
               "initialize completion counter");
    entries.push_back({device, workspace});
    return workspace;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(kWarpMask, value, offset);
    return value;
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

// Each CTA owns four of a weight row's eight original reference warps.
// Each lane keeps its complete stride-256 dot sequence in reference order.
// Only warp totals cross CTAs; the winner adds them in the original order.
// The first four rows also retain the norm's 1024 stride-1024 lane sequences.
template <int Tokens>
__global__ void hc_last_cta(const __nv_bfloat16* __restrict__ x,
                            const float* __restrict__ fn,
                            const float* __restrict__ scale,
                            const float* __restrict__ base,
                            const float* __restrict__ pre_in,
                            __nv_bfloat16* __restrict__ y,
                            float* __restrict__ pre, float* __restrict__ post,
                            float* __restrict__ comb, Workspace* workspace) {
    __shared__ float mixes[Tokens][kHcMix];
    __shared__ float reciprocal_rms[Tokens];
    __shared__ bool last;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.y;
    const int part = blockIdx.x;
    const int original_lane = part * kThreads + tid;
    float dots[Tokens] = {};
    float squares[Tokens] = {};
#pragma unroll 1
    for (int step = 0; step < kStreamSize / kDotThreads; ++step) {
        const int column = original_lane + step * kDotThreads;
        const float weight = fn[row * kStreamSize + column];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const float value = __bfloat162float(x[token * kStreamSize + column]);
            dots[token] += value * weight;
            // For row r=0..3 this selects columns r*256+original_lane,
            // r*256+original_lane+1024, ...: exactly one reference norm lane.
            if (row < kHc && (step & (kHc - 1)) == row)
                squares[token] += value * value;
        }
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float dot = warp_sum(dots[token]);
        if (lane == 0)
            workspace->dots[token][row][part * kWarps + warp] = dot;
        if (row < kHc) {
            const float square = warp_sum(squares[token]);
            if (lane == 0)
                workspace->squares[token][row * kDotWarps + part * kWarps + warp] = square;
        }
    }

    // Every partial writer happens-before this CTA's leader publication.
    __syncthreads();
    if (tid == 0) {
        Completion completed(workspace->completed);
        // Each acq_rel RMW acquires its predecessor in modification order and
        // releases that predecessor's publications plus this CTA's writes.
        // Consequently the final ticket acquires ALL partials transitively.
        const unsigned ticket = completed.fetch_add(1, cuda::memory_order_acq_rel);
        last = ticket == unsigned(kParts * kHcMix - 1);
    }
    // Distribute both the winner flag and the acquired partials to the CTA.
    __syncthreads();
    if (!last) return;  // Uniform: no waiting for another, possibly unscheduled CTA.

    // Sum the eight dot warp totals / 32 norm warp totals in the exact
    // order used by ops::block_sum<256>/<1024>, including the initial zero.
    for (int index = tid; index < Tokens * kReductionRows; index += kThreads) {
        const int token = index / kReductionRows;
        const int reduction_row = index % kReductionRows;
        float sum = 0.0f;
        if (reduction_row == kHcMix) {
#pragma unroll
            for (int w = 0; w < kNormWarps; ++w)
                sum += workspace->squares[token][w];
            reciprocal_rms[token] = rsqrtf(sum / float(kStreamSize) + kNormEps);
        } else {
#pragma unroll
            for (int w = 0; w < kDotWarps; ++w)
                sum += workspace->dots[token][reduction_row][w];
            mixes[token][reduction_row] = sum;
        }
    }
    __syncthreads();
    for (int token = warp; token < Tokens; token += kWarps)
        finish_coefficients(mixes[token], reciprocal_rms[token], scale, base,
                            pre + token * kHc, post + token * kHc,
                            comb + token * kHc * kHc);

    // The winner computes every collapse output, preserving ops::hc_pre's
    // j=0..3 FP32 accumulation and one BF16 rounding. No other CTA writes y.
    for (int out = tid; out < Tokens * kDim; out += kThreads) {
        const int token = out / kDim;
        const int d = out % kDim;
        float collapsed = 0.0f;
#pragma unroll
        for (int j = 0; j < kHc; ++j)
            collapsed += pre_in[token * kHc + j] *
                         __bfloat162float(x[token * kStreamSize + j * kDim + d]);
        y[out] = __float2bfloat16_rn(collapsed);
    }
    // Reset only after ALL output writes and scratch reads. Other CTAs have
    // already published their sole ticket and never access workspace again.
    // The engine serializes same-device K7 calls/replays, so no next call can
    // race this reset. Release also orders this CTA's writes before the zero.
    __syncthreads();
    if (tid == 0)
        Completion(workspace->completed).store(0, cuda::memory_order_release);
}

}  // namespace

void hc_mixes_pre(const __nv_bfloat16* x, int m, const float* fn, const float* scale, const float* base,
                  const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb,
                  cudaStream_t stream) {
    if (m < 1 || m > kMaxTokens) {
        std::fprintf(stderr, "ds41 k7: invalid token count %d (expected 1..8)\n", m);
        std::abort();
    }
    Workspace* workspace = workspace_for_device(stream);
    const dim3 partial_grid(kParts, kHcMix);
#define K7_LAUNCH(TOKENS) \
    case TOKENS: \
        hc_last_cta<TOKENS><<<partial_grid, kThreads, 0, stream>>>( \
            x, fn, scale, base, pre_in, y, pre, post, comb, workspace); \
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
    check_cuda(cudaGetLastError(), "launch last-CTA mixes and collapse");
}

}  // namespace strata::ds41::kernels
