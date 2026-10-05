// K3-07: two query tokens share a CTA and reuse identical window tiles.
// Four heads per token fill the eight N columns of BF16 m16n8k16 MMA.
// Distinct index lists form a block-diagonal probability tile; a common
// window tile is loaded and multiplied once for both tokens. Everything is
// CTA-local, including FP32 online softmax, so capture needs no allocator.
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
constexpr int kHeadTile = 4;
constexpr int kRows = 32;
constexpr int kStride = kDim + 8;
constexpr int kProbStride = kRows + 8;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr unsigned kWarpMask = 0xffffffffu;

template <bool Paired>
struct __align__(32) TileStorage {
    static constexpr int Heads = Paired ? 8 : 4;
    bf16 q[Heads * kStride];
    bf16 kv[kRows * kStride];
    float score[kWarps][Heads][16];
    bf16 p[Heads * kProbStride];
    float maximum[Heads];
    float sum[Heads];
    float rescale[Heads];
    int index[kRows];
    int common;
};
static_assert(sizeof(TileStorage<true>) <= 48 * 1024, "no shared-memory opt-in required");
static_assert(sizeof(TileStorage<false>) <= 48 * 1024, "no shared-memory opt-in required");

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

template <bool Paired, int Partitions>
__device__ __forceinline__ void scores(TileStorage<Paired>& tile, int warp, int lane) {
    constexpr int heads = TileStorage<Paired>::Heads;
    constexpr int key_tiles = kWarps / Partitions;
    const int key_tile = (warp % key_tiles) * 16;
    const int partition = warp / key_tiles;
    float s[4] = {};
#pragma unroll
    for (int dk = 0; dk < kDim / Partitions; dk += 16) {
        const int d = partition * (kDim / Partitions) + dk;
        unsigned a[4], b[2];
        load_a(a, tile.kv + (key_tile + (lane % 16)) * kStride + d + (lane / 16) * 8);
        load_b(b, tile.q + (lane % heads) * kStride + d + ((lane / 8) & 1) * 8);
        mma(s, a, b);
    }
    const int h = (lane & 3) * 2;
    const int r = lane >> 2;
    if (h < heads) {
        tile.score[warp][h][r] = s[0];
        tile.score[warp][h + 1][r] = s[1];
        tile.score[warp][h][r + 8] = s[2];
        tile.score[warp][h + 1][r + 8] = s[3];
    }
}

template <bool Paired, int OutputDim>
__global__ __launch_bounds__(kThreads) void attention_online(
        const bf16* __restrict__ q, const bf16* __restrict__ window,
        const bf16* __restrict__ comp, const int32_t* __restrict__ idx,
        int m, int n_idx, const float* __restrict__ sink, float scale,
        bf16* __restrict__ output) {
    __shared__ TileStorage<Paired> tile;
    constexpr int heads = TileStorage<Paired>::Heads;
    constexpr int queries = Paired ? 2 : 1;
    constexpr int step = kRows / queries;
    constexpr int output_tiles = OutputDim / (kWarps * 16);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int query = blockIdx.z * queries;
    const int head = blockIdx.x * kHeadTile;
    const int output_dim = blockIdx.y * OutputDim;
    const int head0 = (lane & 3) * 2;
    const int head1 = head0 + 1;
    const int mma_row = lane >> 2;

    for (int i = threadIdx.x; i < heads * kDim / 8; i += kThreads) {
        const int h = i / (kDim / 8);
        const int d = (i % (kDim / 8)) * 8;
        const int token = query + h / kHeadTile;
        const size_t offset = (static_cast<size_t>(token) * kHeads + head + h % kHeadTile) * kDim + d;
        *reinterpret_cast<uint4*>(tile.q + h * kStride + d) = token < m
            ? *reinterpret_cast<const uint4*>(q + offset) : make_uint4(0, 0, 0, 0);
    }
    if (threadIdx.x < heads) {
        tile.maximum[threadIdx.x] = -1.0e30f;
        tile.sum[threadIdx.x] = 0.0f;
    }
    float result[output_tiles][4];
#pragma unroll
    for (int v = 0; v < output_tiles; ++v)
#pragma unroll
        for (int e = 0; e < 4; ++e) result[v][e] = 0.0f;
    __syncthreads();

    for (int first = 0; first < n_idx; first += step) {
        if (warp == 0) {
            const int token = query + lane / step;
            const int pos = first + lane % step;
            const int j = token < m && pos < n_idx ? idx[static_cast<size_t>(token) * n_idx + pos] : -1;
            tile.index[lane] = j;
            if constexpr (Paired) {
                // Compare all sixteen positions, including holes. Never infer
                // common window membership from the position in the list.
                const int other = __shfl_xor_sync(kWarpMask, j, 16);
                const bool common = __all_sync(kWarpMask, j == other && j < kWindow);
                if (lane == 0) tile.common = common;
            }
        }
        __syncthreads();
        const bool common = Paired && tile.common;
        const int rows = common ? 16 : kRows;
        for (int i = threadIdx.x; i < rows * kDim / 8; i += kThreads) {
            const int r = i / (kDim / 8);
            const int d = (i % (kDim / 8)) * 8;
            const int j = tile.index[r];
            const bf16* source = window;  // valid base even for zero-fill copies
            if (j >= 0) {
                source = j < kWindow ? window + static_cast<size_t>(j) * kDim
                                     : comp + static_cast<size_t>(j - kWindow) * kDim;
                source += d;
            }
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                         :: "r"(shared_address(tile.kv + r * kStride + d)),
                            "l"(source), "r"(j >= 0 ? 16 : 0) : "memory");
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // Four warps split the reduction dimension. For identical windows
        // all eight head columns use one key tile; otherwise two key tiles
        // hold the independent lists and only their own head columns survive.
        if (common) scores<Paired, 4>(tile, warp, lane);
        else scores<Paired, 2>(tile, warp, lane);
        __syncthreads();

        if (threadIdx.x < heads * 16) {
            const int h = threadIdx.x / 16;
            const int sublane = threadIdx.x % 16;
            float s[2];
            float maximum = tile.maximum[h];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int r = sublane + e * 16;
                const bool owned = !Paired || (common ? e == 0 : e == h / kHeadTile);
                const bool valid = owned && tile.index[r] >= 0;
                float dot = tile.score[e][h][sublane] + tile.score[e + 2][h][sublane];
                if (common)
                    dot = (tile.score[0][h][sublane] + tile.score[1][h][sublane])
                        + (tile.score[2][h][sublane] + tile.score[3][h][sublane]);
                s[e] = valid ? dot * scale : -CUDART_INF_F;
                maximum = fmaxf(maximum, s[e]);
            }
#pragma unroll
            for (int offset = 8; offset; offset >>= 1)
                maximum = fmaxf(maximum, __shfl_xor_sync(kWarpMask, maximum, offset, 16));
            const float alpha = expf(tile.maximum[h] - maximum);
            float sum = 0.0f;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float p = s[e] == -CUDART_INF_F ? 0.0f : expf(s[e] - maximum);
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
        }
        __syncthreads();

        const float alpha0 = tile.rescale[head0 % heads];
        const float alpha1 = tile.rescale[head1 % heads];
#pragma unroll
        for (int v = 0; v < output_tiles; ++v) {
            result[v][0] *= alpha0;
            result[v][1] *= alpha1;
            result[v][2] *= alpha0;
            result[v][3] *= alpha1;
        }
#pragma unroll
        for (int r = 0; r < kRows; r += 16) {
            if (r == 0 || !common) {
                unsigned p[2];
                load_b(p, tile.p + (lane % heads) * kProbStride + r + ((lane / 8) & 1) * 8);
#pragma unroll
                for (int v = 0; v < output_tiles; ++v) {
                    const int d = output_dim + (warp * output_tiles + v) * 16;
                    unsigned values[4];
                    load_a_transposed(values, tile.kv + (r + (lane % 8) + (lane / 16) * 8) * kStride
                                                         + d + ((lane / 8) & 1) * 8);
                    mma(result[v], values, p);
                }
            }
        }
        __syncthreads();  // protect shared KV/P/index until all consumers finish
    }

    if (head0 < heads && query + head0 / kHeadTile < m) {
        const int token = query + head0 / kHeadTile;
        const int h0 = head + head0 % kHeadTile;
        const int h1 = head + head1 % kHeadTile;
        const float denom0 = tile.sum[head0] + expf(sink[h0] - tile.maximum[head0]);
        const float denom1 = tile.sum[head1] + expf(sink[h1] - tile.maximum[head1]);
        bf16* out0 = output + (static_cast<size_t>(token) * kHeads + h0) * kDim;
        bf16* out1 = output + (static_cast<size_t>(token) * kHeads + h1) * kDim;
#pragma unroll
        for (int v = 0; v < output_tiles; ++v) {
            const int d = output_dim + (warp * output_tiles + v) * 16 + mma_row;
            out0[d] = __float2bfloat16_rn(result[v][0] / denom0);
            out1[d] = __float2bfloat16_rn(result[v][1] / denom1);
            out0[d + 8] = __float2bfloat16_rn(result[v][2] / denom0);
            out1[d + 8] = __float2bfloat16_rn(result[v][3] / denom1);
        }
    }
}
}  // namespace

void sparse_attn_decode(const bf16* q, const bf16* window, const bf16* comp,
                        const int32_t* idx, int m, int n_idx, const float* sink, float scale,
                        bf16* o, cudaStream_t stream) {
    if (m <= 0) return;
    if (m > 8 || n_idx < 0 || n_idx > 1024) {
        std::fprintf(stderr, "sparse_attn_decode: invalid shape m=%d n_idx=%d\n", m, n_idx);
        std::abort();
    }
    if (m == 1)
        attention_online<false, 64><<<dim3(kHeads / kHeadTile, kDim / 64, 1), kThreads, 0, stream>>>(
            q, window, comp, idx, m, n_idx, sink, scale, o);
    else
        attention_online<true, 128><<<dim3(kHeads / kHeadTile, kDim / 128, (m + 1) / 2), kThreads, 0, stream>>>(
            q, window, comp, idx, m, n_idx, sink, scale, o);
}
}  // namespace strata::ds41::kernels
