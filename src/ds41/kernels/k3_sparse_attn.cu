// K3-04: allocation-free, two-pass sparse attention with asynchronous gathered
// KV tiles. Two heads reuse each tile; Q stays in registers during the QK pass.
// Output-dimension partitioning supplies decode parallelism without split-KV
// scratch or changing the reference's full-list BF16 probability rounding.
#include "strata/ds41/kernels/k3_sparse_attn.hpp"

#include "strata/ds41/config.hpp"

#include <math_constants.h>
#include <cstdio>
#include <cstdlib>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;
constexpr unsigned kFullWarp = 0xffffffffu;
constexpr int kHeadsPerBlock = 2;
constexpr int kWarpsPerHead = 2;
constexpr int kThreads = kHeadsPerBlock * kWarpsPerHead * 32;
constexpr int kTileRows = 16;
constexpr int kOutputDims = 128;
constexpr int kMaxRows = 1024;

struct __align__(16) SharedStorage {
    bf16 kv[2][kTileRows * kHeadDim];
    float probability[kHeadsPerBlock][kMaxRows];
    float denominator[kHeadsPerBlock];
};
static_assert(sizeof(SharedStorage) == 40976, "shared layout changed");
static_assert(sizeof(SharedStorage) <= 48 * 1024, "no shared-memory opt-in needed");
static_assert(kOutputDims * (kHeadDim / kOutputDims) == kHeadDim, "output tiling");

__device__ __forceinline__ void copy_async_16(bf16* destination, const bf16* source, bool valid) {
    const unsigned shared_address = static_cast<unsigned>(__cvta_generic_to_shared(destination));
    // A zero source-size fills an empty slot without reading from the source.
    // All required targets support cp.async (Ampere or newer).
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(shared_address), "l"(source), "r"(valid ? 16 : 0) : "memory");
}

__device__ __forceinline__ void commit_copies() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void finish_copies() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    // cp.async completion is per thread; this barrier also publishes the
    // other producers' copies and protects the buffer just consumed.
    __syncthreads();
}

template <int Width>
__device__ __forceinline__ void gather_tile(bf16* destination, const bf16* window,
                                           const bf16* comp, const int32_t* indices,
                                           int first, int n_idx, int first_dimension) {
#pragma unroll
    for (int vector = threadIdx.x; vector < kTileRows * Width / 8; vector += kThreads) {
        const int row = vector / (Width / 8);
        const int dim = (vector % (Width / 8)) * 8;
        const int j = first + row < n_idx ? indices[first + row] : -1;
        const bf16* source = window;  // valid pointer even for a zero-fill copy
        if (j >= 0) {
            source = j < kWindow ? window + static_cast<size_t>(j) * kHeadDim
                                 : comp + static_cast<size_t>(j - kWindow) * kHeadDim;
            source += first_dimension + dim;
        }
        copy_async_16(destination + row * Width + dim, source, j >= 0);
    }
    commit_copies();
}

__device__ __forceinline__ float sum_down(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(kFullWarp, value, offset);
    return value;
}

__device__ __forceinline__ float max_all(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(kFullWarp, value, offset));
    return value;
}

// Each CTA owns two heads and 128 output dimensions. Four independent output
// tiles give m=1 a grid of 128 CTAs. Recomputing QK trades some arithmetic for
// this parallelism and avoids allocator overhead and global intermediate data.
// Two warps per head divide score rows in pass 1 and output dimensions in pass 2.
__global__ __launch_bounds__(kThreads) void attention_two_pass(
        const bf16* __restrict__ q, const bf16* __restrict__ window,
        const bf16* __restrict__ comp, const int32_t* __restrict__ idx,
        int n_idx, const float* __restrict__ sink, float scale,
        bf16* __restrict__ output) {
    __shared__ SharedStorage storage;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int local_head = warp % kHeadsPerBlock;
    const int head_warp = warp / kHeadsPerBlock;
    const int head = blockIdx.x * kHeadsPerBlock + local_head;
    const int query = blockIdx.z;
    const int output_dim = blockIdx.y * kOutputDims;
    const int query_head = query * kHeads + head;
    const int32_t* indices = idx + static_cast<size_t>(query) * n_idx;

    float query_values[kHeadDim / 32];
#pragma unroll
    for (int d = 0; d < kHeadDim / 32; ++d)
        query_values[d] = __bfloat162float(q[static_cast<size_t>(query_head) * kHeadDim + d * 32 + lane]);

    if (n_idx > 0) {
        gather_tile<kHeadDim>(storage.kv[0], window, comp, indices, 0, n_idx, 0);
        finish_copies();
    }
    for (int first = 0, buffer = 0; first < n_idx; first += kTileRows, buffer ^= 1) {
        if (first + kTileRows < n_idx)
            gather_tile<kHeadDim>(storage.kv[buffer ^ 1], window, comp, indices,
                                 first + kTileRows, n_idx, 0);
        const bf16* tile = storage.kv[buffer];
#pragma unroll 1
        for (int row = head_warp; row < kTileRows; row += kWarpsPerHead) {
            float score = 0.0f;
            // This lane-strided FP32 reduction is the reference's QK order.
#pragma unroll
            for (int d = 0; d < kHeadDim / 32; ++d)
                score += query_values[d] * __bfloat162float(tile[row * kHeadDim + d * 32 + lane]);
            score = sum_down(score);
            if (lane == 0 && first + row < n_idx)
                storage.probability[local_head][first + row] =
                    indices[first + row] >= 0 ? score * scale : -CUDART_INF_F;
        }
        finish_copies();
    }

    // One warp owns each complete score row. The denominator uses unrounded
    // exponentials; only the PV weights are rounded to BF16. The sink enters
    // the denominator once and does not change the maximum, just as specified.
    if (warp < kHeadsPerBlock) {
        float* scores = storage.probability[warp];
        float maximum = -1.0e30f;
        for (int row = lane; row < n_idx; row += 32) maximum = fmaxf(maximum, scores[row]);
        maximum = max_all(maximum);
        float total = 0.0f;
        for (int row = lane; row < n_idx; row += 32) {
            const float p = scores[row] == -CUDART_INF_F ? 0.0f : expf(scores[row] - maximum);
            total += p;
            scores[row] = __bfloat162float(__float2bfloat16_rn(p));
        }
        total = sum_down(total);
        if (lane == 0)
            storage.denominator[warp] = total + expf(sink[blockIdx.x * kHeadsPerBlock + warp] - maximum);
    }
    __syncthreads();

    // The second pipeline gathers just this CTA's 128 dimensions. A packed
    // BF16 pair per lane avoids the half-word shared-memory bank conflicts.
    const int local_dim = head_warp * 64 + lane * 2;
    float value0 = 0.0f;
    float value1 = 0.0f;
    if (n_idx > 0) {
        gather_tile<kOutputDims>(storage.kv[0], window, comp, indices, 0, n_idx, output_dim);
        finish_copies();
    }
    for (int first = 0, buffer = 0; first < n_idx; first += kTileRows, buffer ^= 1) {
        if (first + kTileRows < n_idx)
            gather_tile<kOutputDims>(storage.kv[buffer ^ 1], window, comp, indices,
                                    first + kTileRows, n_idx, output_dim);
        const bf16* tile = storage.kv[buffer];
#pragma unroll 1
        for (int row = 0; row < kTileRows && first + row < n_idx; ++row) {
            const float p = storage.probability[local_head][first + row];
            const float2 value = __bfloat1622float2(
                *reinterpret_cast<const bf162*>(tile + row * kOutputDims + local_dim));
            value0 += p * value.x;
            value1 += p * value.y;
        }
        finish_copies();
    }
    const float denominator = storage.denominator[local_head];
    const bf162 result = __floats2bfloat162_rn(value0 / denominator, value1 / denominator);
    *reinterpret_cast<bf162*>(output + static_cast<size_t>(query_head) * kHeadDim + output_dim + local_dim) = result;
}
}  // namespace

void sparse_attn_decode(const bf16* q, const bf16* window, const bf16* comp,
                        const int32_t* idx, int m, int n_idx, const float* sink, float scale,
                        bf16* o, cudaStream_t stream) {
    if (m <= 0) return;
    if (n_idx < 0 || n_idx > kMaxRows) {
        std::fprintf(stderr, "sparse_attn_decode: invalid n_idx %d\n", n_idx);
        std::abort();
    }
    attention_two_pass<<<dim3(kHeads / kHeadsPerBlock, kHeadDim / kOutputDims, m), kThreads, 0, stream>>>(
        q, window, comp, idx, n_idx, sink, scale, o);
}
}  // namespace strata::ds41::kernels
