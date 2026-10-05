// src/ds41/kernels/k7_hc.cu - paired-token Sinkhorn with reference-order FP32 mixes.
#include "strata/ds41/kernels/k7_hc.hpp"

#include "strata/ds41/config.hpp"

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

// Each independent 16-lane subgroup owns one token's 4x4 matrix. The
// explicit mask contains only that subgroup, and width=16 makes source lanes
// relative to it. An odd final token never names the absent upper halfwarp.
// Sequential additions reproduce reference parenthesization, not a tree sum.
__device__ __forceinline__ float row_sum(float value, int lane, unsigned mask) {
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kHc; ++k)
        sum += __shfl_sync(mask, value, (lane & 12) + k, 16);
    return sum;
}

__device__ __forceinline__ float column_sum(float value, int lane, unsigned mask) {
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kHc; ++j)
        sum += __shfl_sync(mask, value, j * kHc + (lane & 3), 16);
    return sum;
}

__device__ __forceinline__ void finish_coefficients(
    const float* mixes, float reciprocal_rms, const float* scale, const float* base,
    float* pre, float* post, float* comb) {
    const int lane = threadIdx.x & 15;
    const unsigned mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
    if (lane < kHc) {
        // Preserve the FP32 rounding between normalization and scale/bias.
        const float pm = __fmul_rn(mixes[lane], reciprocal_rms);
        const float qm = __fmul_rn(mixes[lane + kHc], reciprocal_rms);
        pre[lane] = 1.0f / (1.0f + expf(-(pm * scale[0] + base[lane]))) + kHcEps;
        post[lane] = 2.0f / (1.0f + expf(-(qm * scale[1] + base[lane + kHc])));
    }
    const float mix = __fmul_rn(mixes[2 * kHc + lane], reciprocal_rms);
    float c = mix * scale[2] + base[2 * kHc + lane];
    float maximum = -INFINITY;
#pragma unroll
    for (int k = 0; k < kHc; ++k)
        maximum = fmaxf(maximum, __shfl_sync(mask, c, (lane & 12) + k, 16));
    c = expf(c - maximum);
    c = c / row_sum(c, lane, mask) + kHcEps;

    // Exactly the reference: initial column normalization, followed by 19
    // row/column pairs. The epsilon stays in EVERY normalization denominator.
    c = c / (column_sum(c, lane, mask) + kHcEps);
#pragma unroll 1
    for (int iteration = 0; iteration < kSinkhornIters - 1; ++iteration) {
        c = c / (row_sum(c, lane, mask) + kHcEps);
        c = c / (column_sum(c, lane, mask) + kHcEps);
    }
    comb[lane] = c;
}

// Launch ordering on the supplied stream publishes ALL partials before this
// stage. There is no cross-CTA spin, fence, atomic, or in-kernel global barrier.
// Each feature-tile CTA collapses up to two tokens, one at a time; the
// first tile additionally finishes their coefficients in independent halfwarps.
__global__ void hc_finish(const __nv_bfloat16* __restrict__ x, int m,
                          const float* __restrict__ scale, const float* __restrict__ base,
                          const float* __restrict__ pre_in, __nv_bfloat16* __restrict__ y,
                          float* __restrict__ pre, float* __restrict__ post,
                          float* __restrict__ comb, const Workspace* workspace) {
    __shared__ float mixes[2][kHcMix];
    __shared__ float reciprocal_rms[2];
    const int tid = threadIdx.x;
    const int first_token = blockIdx.y * 2;
    const int d = blockIdx.x * kThreads + tid;

    // Match ops::hc_pre's j=0..3 FP32 accumulation and single BF16 rounding.
    // The token guard is uniform across the CTA and precedes all tail accesses.
#pragma unroll
    for (int member = 0; member < 2; ++member) {
        const int token = first_token + member;
        if (token < m) {
            float collapsed = 0.0f;
#pragma unroll
            for (int j = 0; j < kHc; ++j)
                collapsed += pre_in[token * kHc + j] *
                             __bfloat162float(x[token * kStreamSize + j * kDim + d]);
            y[token * kDim + d] = __float2bfloat16_rn(collapsed);
        }
    }
    if (blockIdx.x != 0) return;  // Uniform for the CTA, before any barrier.

    if (tid < 2 * kReductionRows) {
        const int member = tid / kReductionRows;
        const int row = tid % kReductionRows;
        const int token = first_token + member;
        if (token < m) {
            // Start at +0 and sum warp totals in exactly ops::block_sum order.
            float sum = 0.0f;
            if (row == kHcMix) {
#pragma unroll
                for (int w = 0; w < kNormWarps; ++w)
                    sum += workspace->squares[token][w];
                reciprocal_rms[member] = rsqrtf(sum / static_cast<float>(kStreamSize) + kNormEps);
            } else {
#pragma unroll
                for (int w = 0; w < kDotWarps; ++w)
                    sum += workspace->dots[token][row][w];
                mixes[member][row] = sum;
            }
        }
    }
    __syncthreads();
    if (tid < 32) {
        const int member = tid / 16;
        const int token = first_token + member;
        // The whole named subgroup participates, or none of it does. This
        // also prevents reads of the absent token's uninitialized shared data.
        if (token < m)
            finish_coefficients(mixes[member], reciprocal_rms[member], scale, base,
                                pre + token * kHc, post + token * kHc,
                                comb + token * kHc * kHc);
    }
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
    const dim3 partial_grid(kDotWarps, kHcMix);
#define K7_LAUNCH(TOKENS) \
    case TOKENS: \
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
    hc_finish<<<dim3(kDim / kThreads, (m + 1) / 2), kThreads, 0, stream>>>(
        x, m, scale, base, pre_in, y, pre, post, comb, workspace);
    check_cuda(cudaGetLastError(), "launch reduction and collapse");
}

}  // namespace strata::ds41::kernels
