// K3-21: register-resident Q with the K3-11 ping-pong 64-dimension KV slices.
// Eight heads and 128 output dimensions share each CTA. While one 128x64
// buffer feeds QK, cp.async fills the other; the last two slices feed PV.
// XOR-swizzled KV storage needs 32 KiB; the entire CTA uses 37,472 bytes.
// Scores and BF16 probabilities share storage after an explicit read barrier.
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
constexpr int kRows = 128;
constexpr int kSlice = 64;
constexpr int kOutputDim = 128;
constexpr int kProbStride = kRows + 8;
constexpr int kWarps = 4;
constexpr int kThreads = 32 * kWarps;
constexpr int kFragments = kRows / (16 * kWarps);
constexpr int kOutputFragments = kOutputDim / (16 * kWarps);
constexpr unsigned kWarpMask = 0xffffffffu;

struct __align__(32) TileStorage {
    bf16 kv[2][kRows * kSlice];
    union {
        float scores[kHeadTile][kRows];
        bf16 probabilities[kHeadTile * kProbStride];
    } softmax;
    int indices[kRows];
    float maximum[kHeadTile];
    float sum[kHeadTile];
    float rescale[kHeadTile];
};
static_assert(sizeof(TileStorage) == 37472, "shared-memory layout changed");
static_assert(sizeof(TileStorage) <= 48 * 1024, "no shared-memory opt-in required");
static_assert(kRows == kThreads, "one thread gathers each index");

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

// XOR complete 16-byte chunks, leaving the eight BF16 values in each
// chunk contiguous and aligned. Eight adjacent rows use distinct banks.
__device__ __forceinline__ int kv_offset(int row, int dimension) {
    return row * kSlice + (dimension ^ ((row & 7) * 8));
}

// Issue only: the caller overlaps these copies with MMA on the other
// buffer and waits immediately before the destination's first read.
// Negative indices use a safe base with src-size zero, never a bad address.
__device__ __forceinline__ void gather_slice(TileStorage& tile, int buffer,
                                             const bf16* window, const bf16* comp,
                                             int dimension) {
// Keep gather temporaries short-lived while all 64 Q registers are live.
    // Packet addresses, issue order, commit, and waits are unchanged.
#pragma unroll 1
    for (int i = threadIdx.x; i < kRows * kSlice / 8; i += kThreads) {
        const int r = i / (kSlice / 8);
        const int d = (i % (kSlice / 8)) * 8;
        const int j = tile.indices[r];
        const bf16* source = window;
        if (j >= 0) {
            source = j < kWindow ? window + static_cast<size_t>(j) * kDim
                                 : comp + static_cast<size_t>(j - kWindow) * kDim;
            source += dimension + d;
        }
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(shared_address(tile.kv[buffer] + kv_offset(r, d))),
                        "l"(source), "r"(j >= 0 ? 16 : 0) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void wait_slice() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    // Each thread waits for its own writes, then the CTA publishes all
    // completed copies and finishes all readers of the other buffer.
    __syncthreads();
}

// Bound register allocation; the rolled gather avoids spilling Q fragments.
// Shared memory remains the two-CTA residency limit on a 100-KiB-SMEM SM.
__global__ __maxnreg__(112) void attention_online(
        const bf16* __restrict__ q, const bf16* __restrict__ window,
        const bf16* __restrict__ comp, const int32_t* __restrict__ idx,
        int n_idx, const float* __restrict__ sink, float scale,
        bf16* __restrict__ output) {
    __shared__ TileStorage tile;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int query = blockIdx.z;
    const int output_group = blockIdx.y;
    const int output_dimension = output_group * kOutputDim;
    const int head = blockIdx.x * kHeadTile;
    const int head0 = (lane & 3) * 2;
    const int head1 = head0 + 1;
    const int mma_row = lane >> 2;
    const int32_t* indices = idx + static_cast<size_t>(query) * n_idx;
    const bf16* queries = q + (static_cast<size_t>(query) * kHeads + head) * kDim;

    // MMA B ownership: lane 4*h+t holds head h and K pairs 2*t, 2*t+8.
    // Across a warp each of the 8*512 query values has exactly one owner.
    // The four key-owning warps each need the same tile; no lane holds a
    // complete query (128 BF16 values per lane, packed into 64 registers).
    // Pre-rotate fragments to preserve K3-11's cyclic FP32 reduction order.
    const int first_dimension = (output_dimension + kOutputDim) & (kDim - 1);
    unsigned query_fragments[kDim / kSlice][kSlice / 16][2];
#pragma unroll
    for (int slice = 0; slice < kDim / kSlice; ++slice) {
        const int dimension = (first_dimension + slice * kSlice) & (kDim - 1);
#pragma unroll
        for (int d = 0; d < kSlice / 16; ++d) {
            const bf16* pair = queries + (lane / 4) * kDim
                                      + dimension + d * 16 + (lane % 4) * 2;
            query_fragments[slice][d][0] = *reinterpret_cast<const unsigned*>(pair);
            query_fragments[slice][d][1] = *reinterpret_cast<const unsigned*>(pair + 8);
        }
    }
    if (threadIdx.x < kHeadTile) {
        tile.maximum[threadIdx.x] = -1.0e30f;
        tile.sum[threadIdx.x] = 0.0f;
    }
    float result[kOutputFragments][4] = {};
    __syncthreads();

    for (int first = 0; first < n_idx; first += kRows) {
        const int position = first + threadIdx.x;
        tile.indices[threadIdx.x] = position < n_idx ? indices[position] : -1;
        __syncthreads();
        float score[kFragments][4] = {};
        // A cyclic dimension order keeps the same FP32 accumulation order
        // as 128-dimension slices, ending at this CTA's two output slices.
        gather_slice(tile, 0, window, comp, first_dimension);
        wait_slice();
        // Constant fragment indices keep all Q fragments in registers.
#pragma unroll
        for (int slice = 0; slice < kDim / kSlice; ++slice) {
            const int buffer = slice & 1;
            const int dimension = (first_dimension + slice * kSlice) & (kDim - 1);
            if (slice + 1 < kDim / kSlice) {
                // On slice zero the other buffer has no readers. Later,
                // the preceding wait_slice barrier retired every reader
                // of that buffer before this overwrite can be issued.
                const int next_dimension = (dimension + kSlice) & (kDim - 1);
                gather_slice(tile, buffer ^ 1, window, comp, next_dimension);
            }
#pragma unroll
            for (int d = 0; d < kSlice; d += 16) {
#pragma unroll
                for (int v = 0; v < kFragments; ++v) {
                    const int key_tile = (warp * kFragments + v) * 16;
                    unsigned keys[4];
                    load_a(keys, tile.kv[buffer] + kv_offset(key_tile + (lane % 16),
                                                           d + (lane / 16) * 8));
                    mma(score[v], keys, query_fragments[slice][d / 16]);
                }
            }
            // This wait is AFTER current-slice MMA, overlapping all four
            // K=16 steps with the next gather. It publishes the next slice
            // and protects current-buffer reuse with one CTA barrier.
            // The final slice has no outstanding copies; the score barrier
            // below retires its readers without another empty wait.
            if (slice + 1 < kDim / kSlice) wait_slice();
        }
#pragma unroll
        for (int v = 0; v < kFragments; ++v) {
            const int row = (warp * kFragments + v) * 16 + mma_row;
            tile.softmax.scores[head0][row] = score[v][0];
            tile.softmax.scores[head1][row] = score[v][1];
            tile.softmax.scores[head0][row + 8] = score[v][2];
            tile.softmax.scores[head1][row + 8] = score[v][3];
        }
        __syncthreads();

        // Sixteen lanes own one head, with eight keys per lane. All score
        // reads finish before ANY probability write aliases the union.
        const int h = threadIdx.x / 16;
        const int sublane = threadIdx.x % 16;
        float scores[kRows / 16];
        float maximum = tile.maximum[h];
#pragma unroll
        for (int e = 0; e < kRows / 16; ++e) {
            const int r = sublane + e * 16;
            scores[e] = tile.indices[r] >= 0 ? tile.softmax.scores[h][r] * scale : -CUDART_INF_F;
            maximum = fmaxf(maximum, scores[e]);
        }
#pragma unroll
        for (int offset = 8; offset; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(kWarpMask, maximum, offset, 16));
        const float alpha = expf(tile.maximum[h] - maximum);
        __syncthreads();
        float sum = 0.0f;
#pragma unroll
        for (int e = 0; e < kRows / 16; ++e) {
            const float p = scores[e] == -CUDART_INF_F ? 0.0f : expf(scores[e] - maximum);
            sum += p;
            tile.softmax.probabilities[h * kProbStride + sublane + e * 16] = __float2bfloat16_rn(p);
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

        const float alpha0 = tile.rescale[head0];
        const float alpha1 = tile.rescale[head1];
#pragma unroll
        for (int v = 0; v < kOutputFragments; ++v) {
            result[v][0] *= alpha0;
            result[v][1] *= alpha1;
            result[v][2] *= alpha0;
            result[v][3] *= alpha1;
        }
#pragma unroll
        for (int r = 0; r < kRows; r += 16) {
            unsigned probabilities[2];
            load_b(probabilities, tile.softmax.probabilities + (lane % 8) * kProbStride
                                                                   + r + ((lane / 8) & 1) * 8);
#pragma unroll
            for (int v = 0; v < kOutputFragments; ++v) {
                const int d = (warp * kOutputFragments + v) * 16;
                unsigned values[4];
                // The final even/odd slices remain in buffers zero/one.
                load_a_transposed(values, tile.kv[d / kSlice] +
                    kv_offset(r + (lane % 8) + (lane / 16) * 8,
                              (d % kSlice) + ((lane / 8) & 1) * 8));
                mma(result[v], values, probabilities);
            }
        }
        __syncthreads();  // protect KV and P until all warps finish PV
    }

    // Sink enters only the final unrounded denominator. Empty/all-negative
    // lists retain zero numerator and produce zero for every finite sink.
    const float denom0 = tile.sum[head0] + expf(sink[head + head0] - tile.maximum[head0]);
    const float denom1 = tile.sum[head1] + expf(sink[head + head1] - tile.maximum[head1]);
    bf16* out0 = output + (static_cast<size_t>(query) * kHeads + head + head0) * kDim;
    bf16* out1 = output + (static_cast<size_t>(query) * kHeads + head + head1) * kDim;
#pragma unroll
    for (int v = 0; v < kOutputFragments; ++v) {
        const int d = output_dimension + (warp * kOutputFragments + v) * 16 + mma_row;
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
    if (m > 8 || n_idx < 0 || n_idx > 1024) {
        std::fprintf(stderr, "sparse_attn_decode: invalid shape m=%d n_idx=%d\n", m, n_idx);
        std::abort();
    }
    attention_online<<<dim3(kHeads / kHeadTile, kDim / kOutputDim, m), kThreads, 0, stream>>>(
        q, window, comp, idx, n_idx, sink, scale, o);
}
}  // namespace strata::ds41::kernels
