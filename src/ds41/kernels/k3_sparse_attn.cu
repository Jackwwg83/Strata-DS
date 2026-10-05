// K3-05: eight heads share a 32-key BF16 tensor-core tile. Each CTA streams
// 128 output dimensions; four groups expose more decode CTAs and reduce
// live PV registers. FP32 online softmax requires no allocation.
// m16n8k16 uses heads as the eight columns in both QK and transposed PV;
// the documented MMA layout permits direct per-head register rescaling.
#include "strata/ds41/kernels/k3_sparse_attn.hpp"

#include <math_constants.h>
#include <cstdio>
#include <cstdlib>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
constexpr int kHeads = 64;
constexpr int kDim = 512;
constexpr int kWindow = 128;
constexpr int kHeadTile = 8;
constexpr int kRows = 32;
constexpr int kStride = kDim + 8;
constexpr int kProbStride = kRows + 8;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kOutputDim = 128;
constexpr int kOutputTiles = kOutputDim / (kWarps * 16);
constexpr unsigned kWarpMask = 0xffffffffu;

struct __align__(32) TileStorage {
    bf16 q[kHeadTile * kStride];
    bf16 kv[kRows * kStride];
    float score[2][kHeadTile][kRows];
    bf16 p[kHeadTile * kProbStride];
    float maximum[kHeadTile];
    float sum[kHeadTile];
    float rescale[kHeadTile];
};
static_assert(sizeof(TileStorage) == 44384, "shared-memory layout changed");
static_assert(sizeof(TileStorage) <= 48 * 1024, "no shared-memory opt-in required");

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

__device__ __forceinline__ void gather(TileStorage& tile, const bf16* window,
                                       const bf16* comp, const int32_t* indices,
                                       int first, int n_idx) {
#pragma unroll
    for (int i = threadIdx.x; i < kRows * kDim / 8; i += kThreads) {
        const int row = i / (kDim / 8);
        const int dim = (i % (kDim / 8)) * 8;
        const int j = first + row < n_idx ? indices[first + row] : -1;
        const bf16* source = window;  // valid address even for zero-fill copies
        if (j >= 0) {
            source = j < kWindow ? window + static_cast<size_t>(j) * kDim
                                 : comp + static_cast<size_t>(j - kWindow) * kDim;
            source += dim;
        }
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(shared_address(tile.kv + row * kStride + dim)),
                        "l"(source), "r"(j >= 0 ? 16 : 0) : "memory");
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
}

// One CTA owns eight heads and 128 output dimensions. Splitting each 512-term
// QK dot into two 256-term partials uses all four warps; their sum remains
// FP32. Four output groups repeat QK to expose 32 CTAs per query while
// reducing live PV accumulators. No global score buffer or allocator is used.
__global__ __launch_bounds__(kThreads) void attention_online(
        const bf16* __restrict__ q, const bf16* __restrict__ window,
        const bf16* __restrict__ comp, const int32_t* __restrict__ idx,
        int n_idx, const float* __restrict__ sink, float scale,
        bf16* __restrict__ output) {
    __shared__ TileStorage tile;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int query = blockIdx.z;
    const int output_dim = blockIdx.y * kOutputDim;
    const int head = blockIdx.x * kHeadTile;
    const int32_t* indices = idx + static_cast<size_t>(query) * n_idx;
    const bf16* queries = q + (static_cast<size_t>(query) * kHeads + head) * kDim;
    const int head0 = (lane & 3) * 2;
    const int head1 = head0 + 1;
    const int mma_row = lane >> 2;

    for (int i = threadIdx.x; i < kHeadTile * kDim / 8; i += kThreads) {
        const int h = i / (kDim / 8);
        const int d = (i % (kDim / 8)) * 8;
        *reinterpret_cast<uint4*>(tile.q + h * kStride + d) =
            *reinterpret_cast<const uint4*>(queries + h * kDim + d);
    }
    if (threadIdx.x < kHeadTile) {
        tile.maximum[threadIdx.x] = -1.0e30f;
        tile.sum[threadIdx.x] = 0.0f;
    }
    float result[kOutputTiles][4];
#pragma unroll
    for (int v = 0; v < kOutputTiles; ++v)
#pragma unroll
        for (int e = 0; e < 4; ++e) result[v][e] = 0.0f;
    __syncthreads();

    for (int first = 0; first < n_idx; first += kRows) {
        gather(tile, window, comp, indices, first, n_idx);
        const int partition = warp / 2;
        const int key_tile = (warp % 2) * 16;
        float score[4] = {};
#pragma unroll
        for (int dk = 0; dk < kDim / 2; dk += 16) {
            const int d = partition * (kDim / 2) + dk;
            unsigned a[4], b[2];
            load_a(a, tile.kv + (key_tile + (lane % 16)) * kStride + d + (lane / 16) * 8);
            load_b(b, tile.q + (lane % 8) * kStride + d + ((lane / 8) & 1) * 8);
            mma(score, a, b);
        }
        tile.score[partition][head0][key_tile + mma_row] = score[0];
        tile.score[partition][head1][key_tile + mma_row] = score[1];
        tile.score[partition][head0][key_tile + mma_row + 8] = score[2];
        tile.score[partition][head1][key_tile + mma_row + 8] = score[3];
        __syncthreads();

        // Sixteen lanes cooperate on each head's 32 probabilities. Invalid
        // slots contribute neither numerator nor denominator. Sink is added
        // only once, after all tiles, and does not enter the running maximum.
        const int h = threadIdx.x / 16;
        const int sublane = threadIdx.x % 16;
        float scores[2];
        float maximum = tile.maximum[h];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int r = sublane + e * 16;
            const bool valid = first + r < n_idx && indices[first + r] >= 0;
            scores[e] = valid ? (tile.score[0][h][r] + tile.score[1][h][r]) * scale : -CUDART_INF_F;
            maximum = fmaxf(maximum, scores[e]);
        }
#pragma unroll
        for (int offset = 8; offset; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(kWarpMask, maximum, offset, 16));
        const float alpha = expf(tile.maximum[h] - maximum);
        float sum = 0.0f;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const float p = scores[e] == -CUDART_INF_F ? 0.0f : expf(scores[e] - maximum);
            sum += p;
            tile.p[h * kProbStride + sublane + e * 16] = __float2bfloat16_rn(p);
        }
#pragma unroll
        for (int offset = 8; offset; offset >>= 1)
            sum += __shfl_xor_sync(kWarpMask, sum, offset, 16);
        if (sublane == 0) {
            tile.maximum[h] = maximum;
            tile.sum[h] = tile.sum[h] * alpha + sum;
            tile.rescale[h] = alpha;
        }
        __syncthreads();

        // Output fragments have dimensions on M and heads on N. Every
        // thread therefore always owns two known heads, even across tiles.
        // Rescale directly in registers, with no accumulator staging traffic.
        const float alpha0 = tile.rescale[head0];
        const float alpha1 = tile.rescale[head1];
#pragma unroll
        for (int v = 0; v < kOutputTiles; ++v) {
            result[v][0] *= alpha0;
            result[v][1] *= alpha1;
            result[v][2] *= alpha0;
            result[v][3] *= alpha1;
        }
#pragma unroll
        for (int r = 0; r < kRows; r += 16) {
            unsigned p[2];
            load_b(p, tile.p + (lane % 8) * kProbStride + r + ((lane / 8) & 1) * 8);
#pragma unroll
            for (int v = 0; v < kOutputTiles; ++v) {
                const int d = output_dim + (warp * kOutputTiles + v) * 16;
                unsigned values[4];
                // Transposing the four 8x8 blocks also swaps their order:
                // (key,dim) quadrants are (0,0), (0,8), (8,0), (8,8).
                load_a_transposed(values, tile.kv + (r + (lane % 8) + (lane / 16) * 8) * kStride
                                                     + d + ((lane / 8) & 1) * 8);
                mma(result[v], values, p);
            }
        }
        __syncthreads();  // protect KV/P until every consuming warp is done
    }

    const float denom0 = tile.sum[head0] + expf(sink[head + head0] - tile.maximum[head0]);
    const float denom1 = tile.sum[head1] + expf(sink[head + head1] - tile.maximum[head1]);
    bf16* out0 = output + (static_cast<size_t>(query) * kHeads + head + head0) * kDim;
    bf16* out1 = output + (static_cast<size_t>(query) * kHeads + head + head1) * kDim;
#pragma unroll
    for (int v = 0; v < kOutputTiles; ++v) {
        const int d = output_dim + (warp * kOutputTiles + v) * 16 + mma_row;
        out0[d] = __float2bfloat16_rn(result[v][0] / denom0);
        out1[d] = __float2bfloat16_rn(result[v][1] / denom1);
        out0[d + 8] = __float2bfloat16_rn(result[v][2] / denom0);
        out1[d + 8] = __float2bfloat16_rn(result[v][3] / denom1);
    }
}
}  // namespace

void sparse_attn_decode(const bf16* q, const bf16* window, const bf16* comp,
                        const int32_t* idx, int m, int n_idx, const float* sink, float scale,
                        bf16* o, cudaStream_t stream) {
    if (m <= 0) return;
    if (n_idx < 0 || n_idx > 1024) {
        std::fprintf(stderr, "sparse_attn_decode: invalid n_idx %d\n", n_idx);
        std::abort();
    }
    attention_online<<<dim3(kHeads / kHeadTile, kDim / kOutputDim, m), kThreads, 0, stream>>>(
        q, window, comp, idx, n_idx, sink, scale, o);
}
}  // namespace strata::ds41::kernels
