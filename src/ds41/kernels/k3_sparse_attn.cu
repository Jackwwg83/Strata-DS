// K3-08: 32 heads share each 16-key BF16 tensor-core tile. Two warps
// split each eight-head QK tile into 256-channel reductions. Register-Q
// fragments are half as large and the serial tensor-core chain is halved.
// Output groups expose CTAs; double-buffered KV overlaps gather with MMA.
// Online FP32 softmax rounds P to BF16 before PV. No global scratch/state.
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
constexpr int kHeadTile = 32;
constexpr int kRows = 16;
constexpr int kStride = kDim + 8;
constexpr int kProbStride = kRows + 8;
constexpr int kPartitions = 2;
constexpr int kScoreStride = kRows + 4;
constexpr int kWarps = 8;
constexpr int kThreads = 32 * kWarps;
constexpr unsigned kWarpMask = 0xffffffffu;

struct __align__(32) TileStorage {
    bf16 kv[2][kRows * kStride];
    bf16 p[kPartitions * kHeadTile * kProbStride];
    float score[kPartitions][kHeadTile * kScoreStride];
};
static_assert(sizeof(TileStorage) == 41472, "shared-memory layout changed");
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

// Each producer writes one disjoint 16-byte packet. A negative index or
// tail packet is zero-filled without forming an address from that index.
__device__ __forceinline__ void prefetch(bf16* dst, const bf16* window,
                                        const bf16* comp, const int32_t* idx,
                                        int first, int n_idx) {
#pragma unroll
    for (int i = threadIdx.x; i < kRows * kDim / 8; i += kThreads) {
        const int row = i / (kDim / 8);
        const int dim = (i % (kDim / 8)) * 8;
        const int j = first + row < n_idx ? idx[first + row] : -1;
        const bf16* source = window;
        if (j >= 0) {
            source = j < kWindow ? window + static_cast<size_t>(j) * kDim
                                 : comp + static_cast<size_t>(j - kWindow) * kDim;
            source += dim;
        }
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(shared_address(dst + row * kStride + dim)),
                        "l"(source), "r"(j >= 0 ? 16 : 0) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// MMA columns are heads in both QK and V^T P^T. Each lane owns two heads
// and two key/output rows separated by eight. The xor-4/8/16 reductions
// therefore leave each head's online softmax state in its owning lanes.
template <int OutputDim>
__global__ __launch_bounds__(kThreads, 2) void attention_online(
        const bf16* __restrict__ q, const bf16* __restrict__ window,
        const bf16* __restrict__ comp, const int32_t* __restrict__ idx,
        int n_idx, const float* __restrict__ sink, float scale,
        bf16* __restrict__ output) {
    __shared__ TileStorage tile;
    constexpr int output_tiles = (OutputDim / kPartitions + 15) / 16;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int query = blockIdx.z;
    const int partition = warp / 4;
    const int head_group = warp % 4;
    const int head = blockIdx.x * kHeadTile + head_group * 8;
    const int output_dim = blockIdx.y * OutputDim;
    const int head0 = (lane & 3) * 2;
    const int head1 = head0 + 1;
    const int row = lane >> 2;
    const int32_t* indices = idx + static_cast<size_t>(query) * n_idx;
    const bf16* queries = q + (static_cast<size_t>(query) * kHeads + head) * kDim;

    // Directly load the documented m16n8k16 B fragment: lane/4 is the
    // column, lane%4 selects a pair in K, and register 1 advances K by 8.
    // Each partition retains only half the query, then reuses it for all keys.
    unsigned query_fragment[kDim / kPartitions / 16][2];
#pragma unroll
    for (int k = 0; k < kDim / kPartitions / 16; ++k) {
        const bf16* src = queries + row * kDim + partition * (kDim / kPartitions) + k * 16 + (lane & 3) * 2;
        query_fragment[k][0] = *reinterpret_cast<const unsigned*>(src);
        query_fragment[k][1] = *reinterpret_cast<const unsigned*>(src + 8);
    }
    float maximum0 = -1.0e30f, maximum1 = -1.0e30f;
    float sum0 = 0.0f, sum1 = 0.0f;
    float result[output_tiles][4] = {};

    if (n_idx > 0) {
        prefetch(tile.kv[0], window, comp, indices, 0, n_idx);
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
    }
    for (int first = 0; first < n_idx; first += kRows) {
        const int buffer = (first / kRows) & 1;
        bf16* kv = tile.kv[buffer];
        if (first + kRows < n_idx)
            prefetch(tile.kv[buffer ^ 1], window, comp, indices, first + kRows, n_idx);

        float score[4] = {};
#pragma unroll
        for (int k = 0; k < kDim / kPartitions / 16; ++k) {
            unsigned keys[4];
            load_a(keys, kv + (lane % 16) * kStride + partition * (kDim / kPartitions) + k * 16 + (lane / 16) * 8);
            mma(score, keys, query_fragment[k]);
        }
        // Adjacent head pairs use a 20-float score stride: the 32 lanes
        // hit all 32 banks instead of colliding across the eight key rows.
        const int sh0 = (head_group * 8 + head0) * kScoreStride + row;
        const int sh1 = (head_group * 8 + head1) * kScoreStride + row;
        tile.score[partition][sh0] = score[0];
        tile.score[partition][sh1] = score[1];
        tile.score[partition][sh0 + 8] = score[2];
        tile.score[partition][sh1 + 8] = score[3];
        __syncthreads();
        score[0] = tile.score[0][sh0] + tile.score[1][sh0];
        score[1] = tile.score[0][sh1] + tile.score[1][sh1];
        score[2] = tile.score[0][sh0 + 8] + tile.score[1][sh0 + 8];
        score[3] = tile.score[0][sh1 + 8] + tile.score[1][sh1 + 8];
        const bool valid0 = first + row < n_idx && indices[first + row] >= 0;
        const bool valid1 = first + row + 8 < n_idx && indices[first + row + 8] >= 0;
        score[0] = valid0 ? score[0] * scale : -CUDART_INF_F;
        score[1] = valid0 ? score[1] * scale : -CUDART_INF_F;
        score[2] = valid1 ? score[2] * scale : -CUDART_INF_F;
        score[3] = valid1 ? score[3] * scale : -CUDART_INF_F;
        float next0 = fmaxf(maximum0, fmaxf(score[0], score[2]));
        float next1 = fmaxf(maximum1, fmaxf(score[1], score[3]));
#pragma unroll
        for (int offset = 4; offset <= 16; offset *= 2) {
            next0 = fmaxf(next0, __shfl_xor_sync(kWarpMask, next0, offset));
            next1 = fmaxf(next1, __shfl_xor_sync(kWarpMask, next1, offset));
        }
        const float alpha0 = expf(maximum0 - next0);
        const float alpha1 = expf(maximum1 - next1);
        const float p0 = valid0 ? expf(score[0] - next0) : 0.0f;
        const float p1 = valid0 ? expf(score[1] - next1) : 0.0f;
        const float p2 = valid1 ? expf(score[2] - next0) : 0.0f;
        const float p3 = valid1 ? expf(score[3] - next1) : 0.0f;
        float add0 = p0 + p2, add1 = p1 + p3;
#pragma unroll
        for (int offset = 4; offset <= 16; offset *= 2) {
            add0 += __shfl_xor_sync(kWarpMask, add0, offset);
            add1 += __shfl_xor_sync(kWarpMask, add1, offset);
        }
        maximum0 = next0;
        maximum1 = next1;
        sum0 = sum0 * alpha0 + add0;
        sum1 = sum1 * alpha1 + add1;

        // P is rounded, while its contribution to the denominator stays
        // FP32. Separate P storage per warp avoids cross-warp softmax barriers.
        bf16* probabilities = tile.p + warp * 8 * kProbStride;
        probabilities[head0 * kProbStride + row] = __float2bfloat16_rn(p0);
        probabilities[head1 * kProbStride + row] = __float2bfloat16_rn(p1);
        probabilities[head0 * kProbStride + row + 8] = __float2bfloat16_rn(p2);
        probabilities[head1 * kProbStride + row + 8] = __float2bfloat16_rn(p3);
        __syncwarp(kWarpMask);
        unsigned p[2];
        load_b(p, probabilities + (lane % 8) * kProbStride + ((lane / 8) & 1) * 8);
#pragma unroll
        for (int v = 0; v < output_tiles; ++v) {
            result[v][0] *= alpha0;
            result[v][1] *= alpha1;
            result[v][2] *= alpha0;
            result[v][3] *= alpha1;
            unsigned values[4];
            const int d = output_dim + (OutputDim >= 32 ? partition * (OutputDim / kPartitions) : 0) + v * 16;
            load_a_transposed(values, kv + ((lane % 8) + (lane / 16) * 8) * kStride
                                             + d + ((lane / 8) & 1) * 8);
            mma(result[v], values, p);
        }
        // All warps finish the current KV/P tile before its buffer can be
        // reused. The same barrier makes the prefetched next tile visible.
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
    }

    // Sink belongs only in the final denominator, never in the running max.
    // An empty list/all-negative list has zero numerator and infinite
    // denominator for finite sink, giving the reference's exact zero output.
    const float denom0 = sum0 + expf(sink[head + head0] - maximum0);
    const float denom1 = sum1 + expf(sink[head + head1] - maximum1);
    bf16* out0 = output + (static_cast<size_t>(query) * kHeads + head + head0) * kDim;
    bf16* out1 = output + (static_cast<size_t>(query) * kHeads + head + head1) * kDim;
#pragma unroll
    for (int v = 0; v < output_tiles; ++v) {
        const int d = output_dim + (OutputDim >= 32 ? partition * (OutputDim / kPartitions) : 0) + v * 16 + row;
        // The narrowest group shares one 16-row PV tile between its two
        // partitions. Each writes its own eight rows; no duplicate stores.
        if (OutputDim >= 32 || partition == 0) {
            out0[d] = __float2bfloat16_rn(result[v][0] / denom0);
            out1[d] = __float2bfloat16_rn(result[v][1] / denom1);
        }
        if (OutputDim >= 32 || partition == 1) {
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
    // Larger query batches supply their own parallelism and can amortize
    // repeated QK across wider output groups. The smallest batch exposes
    // 64 independent CTAs; other batches expose at least 80.
    if (m <= 2)
        attention_online<16><<<dim3(kHeads / kHeadTile, kDim / 16, m), kThreads, 0, stream>>>(
            q, window, comp, idx, n_idx, sink, scale, o);
    else if (m <= 4)
        attention_online<32><<<dim3(kHeads / kHeadTile, kDim / 32, m), kThreads, 0, stream>>>(
            q, window, comp, idx, n_idx, sink, scale, o);
    else
        attention_online<64><<<dim3(kHeads / kHeadTile, kDim / 64, m), kThreads, 0, stream>>>(
            q, window, comp, idx, n_idx, sink, scale, o);
}
}  // namespace strata::ds41::kernels
