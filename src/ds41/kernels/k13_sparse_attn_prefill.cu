// K13-03: tensor-core attention with recomputed logits and no global scratch.
// The first QK pass finds each head's global maximum and unrounded denominator.
// A second QK pass rounds exp(score - GLOBAL maximum) to BF16, then immediately
// accumulates PV. Only one 64-row score tile is live; no online rescaling of a
// rounded probability or partial numerator can change the reference's math.
#include "strata/ds41/kernels/k13_sparse_attn_prefill.hpp"

#include <math_constants.h>
#include <cstdio>
#include <cstdlib>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
constexpr int kHeads = 64;
constexpr int kDim = 512;
constexpr int kHeadTile = 16;
constexpr int kRows = 64;
constexpr int kSlice = 64;
constexpr int kSlices = kDim / kSlice;
constexpr int kThreads = 256;
constexpr int kQueryStride = kDim + 8;
constexpr int kProbabilityStride = kRows + 8;
constexpr unsigned kMask = 0xffffffffu;

struct __align__(32) TileStorage {
    bf16 q[kHeadTile * kQueryStride];
    bf16 kv[kRows * kSlice];
    union {
        float scores[kHeadTile][kRows];
        bf16 probabilities[kHeadTile * kProbabilityStride];
    } softmax;
    const bf16* sources[kRows];
    int valid[kRows];
    float maximum[kHeadTile];
    float sum[kHeadTile];
};
static_assert(sizeof(TileStorage) == 29824, "shared layout changed");
static_assert(sizeof(TileStorage) <= 48 * 1024, "no shared-memory opt-in needed");
static_assert(kHeadTile * 16 == kThreads, "sixteen lanes reduce each head");

__device__ __forceinline__ unsigned shared_address(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void load_a(unsigned (&a)[4], const bf16* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(shared_address(p)) : "memory");
}

__device__ __forceinline__ void load_a_transposed(unsigned (&a)[4], const bf16* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(shared_address(p)) : "memory");
}

__device__ __forceinline__ void load_b(unsigned (&b)[2], const bf16* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(shared_address(p)) : "memory");
}

__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Swizzle 16-byte chunks, preserving aligned vector copies and ldmatrix rows.
__device__ __forceinline__ int kv_offset(int row, int d) {
    return row * kSlice + (d ^ ((row & 7) * 8));
}

__device__ __forceinline__ void select_rows(TileStorage& tile, const bf16* kv,
                                           const int32_t* indices, int first, int n_idx) {
    if (threadIdx.x < kRows) {
        const int p = first + threadIdx.x;
        const int j = p < n_idx ? indices[p] : -1;
        // Invalid slots use an in-bounds base with cp.async source size zero.
        // Cast before multiplication: a valid row may have a >2 GiB offset.
        tile.sources[threadIdx.x] = j >= 0 ? kv + static_cast<size_t>(j) * kDim : kv;
        tile.valid[threadIdx.x] = j >= 0;
    }
    __syncthreads();
}

// One buffer, deliberately: 29,824 B permits three resident CTAs within the
// 99 KiB limit. Two barriers bracket each overwrite, with an async wait between
// the issue and read barrier. There is no pipelined buffer-lifetime assumption.
__device__ __forceinline__ void gather_slice(TileStorage& tile, int dimension) {
    __syncthreads();  // retire every reader of the previous contents
#pragma unroll
    for (int i = threadIdx.x; i < kRows * kSlice / 8; i += kThreads) {
        const int row = i / (kSlice / 8);
        const int d = (i % (kSlice / 8)) * 8;
        const int valid = tile.valid[row];
        const bf16* source = tile.sources[row];
        if (valid) source += dimension + d;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(shared_address(tile.kv + kv_offset(row, d))),
                        "l"(source), "r"(valid ? 16 : 0) : "memory");
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();  // publish all per-thread completed copies to every warp
}

// Each warp owns a 16-key x 8-head product. The first and second passes use
// precisely the same dimension order and MMA instructions, so recomputation
// produces the same logits before BF16 probability rounding.
__device__ __forceinline__ void compute_scores(TileStorage& tile, int warp, int lane) {
    const int head_group = (warp / 4) * 8;
    const int row_group = (warp % 4) * 16;
    float score[4] = {};
    for (int dimension = 0; dimension < kDim; dimension += kSlice) {
        gather_slice(tile, dimension);
#pragma unroll
        for (int d = 0; d < kSlice; d += 16) {
            unsigned a[4], b[2];
            load_a(a, tile.kv + kv_offset(row_group + (lane % 16), d + (lane / 16) * 8));
            load_b(b, tile.q + (head_group + lane % 8) * kQueryStride
                                    + dimension + d + ((lane / 8) & 1) * 8);
            mma(score, a, b);
        }
    }
    const int h = head_group + (lane & 3) * 2;
    const int r = row_group + (lane >> 2);
    tile.softmax.scores[h][r] = score[0];
    tile.softmax.scores[h + 1][r] = score[1];
    tile.softmax.scores[h][r + 8] = score[2];
    tile.softmax.scores[h + 1][r + 8] = score[3];
    __syncthreads();
}

__global__ __launch_bounds__(kThreads, 3) void attention_recompute(
        const bf16* __restrict__ q, const bf16* __restrict__ kv,
        const int32_t* __restrict__ idx, int n_idx,
        const float* __restrict__ sink, float scale, bf16* __restrict__ output) {
    __shared__ TileStorage tile;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int query = blockIdx.x;
    const int head = blockIdx.y * kHeadTile;
    const int32_t* indices = idx + static_cast<size_t>(query) * n_idx;
    const bf16* queries = q + (static_cast<size_t>(query) * kHeads + head) * kDim;
    for (int i = threadIdx.x; i < kHeadTile * kDim / 8; i += kThreads) {
        const int h = i / (kDim / 8);
        const int d = (i % (kDim / 8)) * 8;
        *reinterpret_cast<uint4*>(tile.q + h * kQueryStride + d) =
            *reinterpret_cast<const uint4*>(queries + h * kDim + d);
    }
    const int h = threadIdx.x / 16;
    const int sublane = threadIdx.x % 16;
    float maximum = -1.0e30f;
    float sum = 0.0f;
    __syncthreads();

    // Online accumulation is used ONLY for unrounded normalization statistics.
    // Rounded probabilities and PV never use an intermediate tile maximum.
    for (int first = 0; first < n_idx; first += kRows) {
        select_rows(tile, kv, indices, first, n_idx);
        compute_scores(tile, warp, lane);
        float scores[kRows / 16];
        float next_maximum = maximum;
#pragma unroll
        for (int e = 0; e < kRows / 16; ++e) {
            const int r = sublane + e * 16;
            scores[e] = tile.valid[r] ? tile.softmax.scores[h][r] * scale : -CUDART_INF_F;
            next_maximum = fmaxf(next_maximum, scores[e]);
        }
#pragma unroll
        for (int off = 8; off; off >>= 1)
            next_maximum = fmaxf(next_maximum, __shfl_xor_sync(kMask, next_maximum, off, 16));
        float tile_sum = 0.0f;
#pragma unroll
        for (int e = 0; e < kRows / 16; ++e)
            tile_sum += scores[e] == -CUDART_INF_F ? 0.0f : expf(scores[e] - next_maximum);
#pragma unroll
        for (int off = 8; off; off >>= 1)
            tile_sum += __shfl_xor_sync(kMask, tile_sum, off, 16);
        sum = sum * expf(maximum - next_maximum) + tile_sum;
        maximum = next_maximum;
        __syncthreads();  // retire score and validity readers before next tile
    }
    if (sublane == 0) {
        tile.maximum[h] = maximum;
        // Keep the reference's -1e30 maximum floor and denominator-only sink.
        tile.sum[h] = sum + expf(sink[head + h] - maximum);
    }
    __syncthreads();

    float result[kSlices][4] = {};
    for (int first = 0; first < n_idx; first += kRows) {
        select_rows(tile, kv, indices, first, n_idx);
        compute_scores(tile, warp, lane);
        float scores[kRows / 16];
#pragma unroll
        for (int e = 0; e < kRows / 16; ++e) {
            const int r = sublane + e * 16;
            scores[e] = tile.valid[r] ? tile.softmax.scores[h][r] * scale : -CUDART_INF_F;
        }
        __syncthreads();  // ALL score reads finish before the aliased BF16 writes
#pragma unroll
        for (int e = 0; e < kRows / 16; ++e) {
            const float p = scores[e] == -CUDART_INF_F ? 0.0f : expf(scores[e] - tile.maximum[h]);
            tile.softmax.probabilities[h * kProbabilityStride + sublane + e * 16] = __float2bfloat16_rn(p);
        }
        __syncthreads();

        // The last QK slice is still resident. Consume it first; subsequent
        // gathers visit 0..384. Every output dimension sees key tiles in the
        // same ascending order. All eight warps work on each PV slice.
#pragma unroll
        for (int s = 0; s < kSlices; ++s) {
            constexpr int last = kSlices - 1;
            const int slice = s == 0 ? last : s - 1;
            if (s != 0) gather_slice(tile, slice * kSlice);
#pragma unroll
            for (int r = 0; r < kRows; r += 16) {
                unsigned a[4], b[2];
                load_a_transposed(a, tile.kv + kv_offset(r + lane % 8 + (lane / 16) * 8,
                                                        (warp % 4) * 16 + ((lane / 8) & 1) * 8));
                load_b(b, tile.softmax.probabilities + ((warp / 4) * 8 + lane % 8) * kProbabilityStride
                                                        + r + ((lane / 8) & 1) * 8);
                mma(result[slice], a, b);
            }
        }
        __syncthreads();  // retire KV, probability, and validity readers
    }

    const int h0 = (warp / 4) * 8 + (lane & 3) * 2;
    const int h1 = h0 + 1;
    bf16* out0 = output + (static_cast<size_t>(query) * kHeads + head + h0) * kDim;
    bf16* out1 = output + (static_cast<size_t>(query) * kHeads + head + h1) * kDim;
#pragma unroll
    for (int s = 0; s < kSlices; ++s) {
        const int d = s * kSlice + (warp % 4) * 16 + (lane >> 2);
        out0[d] = __float2bfloat16_rn(result[s][0] / tile.sum[h0]);
        out1[d] = __float2bfloat16_rn(result[s][1] / tile.sum[h1]);
        out0[d + 8] = __float2bfloat16_rn(result[s][2] / tile.sum[h0]);
        out1[d + 8] = __float2bfloat16_rn(result[s][3] / tile.sum[h1]);
    }
}
}  // namespace

void sparse_attn_prefill(const bf16* q, const bf16* kv, const int32_t* idx, int m, int n_idx,
                         const float* sink, float scale, bf16* o, cudaStream_t stream) {
    if (m <= 0) return;
    if (m > 16384 || n_idx < 0 || n_idx > 1024) {
        std::fprintf(stderr, "sparse_attn_prefill: invalid shape m=%d n_idx=%d\n", m, n_idx);
        std::abort();
    }
    attention_recompute<<<dim3(m, kHeads / kHeadTile), kThreads, 0, stream>>>(
        q, kv, idx, n_idx, sink, scale, o);
}
}  // namespace strata::ds41::kernels
