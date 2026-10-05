// Fused tensor-core sparse attention: 16 heads share a 64-row KV tile.
// Four independent 128-wide output groups expose single-query parallelism
// without global temporary storage. FP32 online softmax carries the output
// between KV tiles; only the current probability tile is rounded to BF16.
#include "strata/ds41/kernels/k3_sparse_attn.hpp"

#include <mma.h>
#include <math_constants.h>

#include <cstdio>
#include <cstdlib>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;
constexpr int kHeads = 64;
constexpr int kDim = 512;
constexpr int kWindow = 128;
constexpr int kOutputDim = 128;
constexpr int kHeadTile = 16;
constexpr int kRows = 64;
constexpr int kStride = kDim + 8;
constexpr int kProbStride = kRows + 8;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kOutputsPerWarp = kOutputDim / (16 * kWarps);

struct __align__(32) TileStorage {
    bf16 q[kHeadTile * kStride];
    bf16 kv[kRows * kStride];
    float scores[kHeadTile * kRows];
    bf16 p[kHeadTile * kProbStride];
    // Warp-private staging makes row-dependent rescaling independent of the
    // opaque WMMA accumulator register layout on sm_86, sm_89, and sm_120.
    float stage[kWarps * 16 * 16];
    float maximum[kHeadTile];
    float sum[kHeadTile];
    float rescale[kHeadTile];
};
static_assert(sizeof(TileStorage) == 97984, "shared-memory layout changed");
static_assert(sizeof(TileStorage) <= 99 * 1024, "consumer GPU shared-memory limit");

__device__ __forceinline__ const bf16* kv_row(const bf16* window, const bf16* comp, int j) {
    return j < kWindow ? window + static_cast<size_t>(j) * kDim
                       : comp + static_cast<size_t>(j - kWindow) * kDim;
}

// A block handles one query, 16 heads, and 128 output dimensions. Loading
// each selected KV row serves both QK and PV for every head in the block.
// The four output groups repeat QK to trade compute for occupancy. No global score
// matrix, temporary allocation, inter-kernel softmax, or default-stream work.
__global__ void fused_online(const bf16* __restrict__ q, const bf16* __restrict__ window,
                              const bf16* __restrict__ comp, const int32_t* __restrict__ idx,
                              int n_idx, const float* __restrict__ sink, float scale,
                              bf16* __restrict__ output) {
    extern __shared__ __align__(32) unsigned char shared[];
    TileStorage& tile = *reinterpret_cast<TileStorage*>(shared);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int query = blockIdx.z;
    const int output_dim = blockIdx.y * kOutputDim;
    const int head = blockIdx.x * kHeadTile;
    const bf16* qbase = q + (static_cast<size_t>(query) * kHeads + head) * kDim;
    const int32_t* indices = idx + static_cast<size_t>(query) * n_idx;
    float* stage = tile.stage + warp * 16 * 16;
    for (int i = threadIdx.x; i < kHeadTile * kDim / 8; i += kThreads) {
        const int h = i / (kDim / 8);
        const int d = (i % (kDim / 8)) * 8;
        *reinterpret_cast<uint4*>(tile.q + h * kStride + d) =
            *reinterpret_cast<const uint4*>(qbase + h * kDim + d);
    }
    if (threadIdx.x < kHeadTile) {
        tile.maximum[threadIdx.x] = -1.0e30f;
        tile.sum[threadIdx.x] = 0.0f;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> out[kOutputsPerWarp];
#pragma unroll
    for (int v = 0; v < kOutputsPerWarp; ++v) wmma::fill_fragment(out[v], 0.0f);
    __syncthreads();

    for (int first = 0; first < n_idx; first += kRows) {
        for (int i = threadIdx.x; i < kRows * kDim / 8; i += kThreads) {
            const int r = i / (kDim / 8);
            const int d = (i % (kDim / 8)) * 8;
            const int j = first + r < n_idx ? indices[first + r] : -1;
            const uint4 values = j >= 0 ? *reinterpret_cast<const uint4*>(kv_row(window, comp, j) + d)
                                       : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(tile.kv + r * kStride + d) = values;
        }
        __syncthreads();
        // Four warps cover the 16x64 score matrix, keeping all 512-term dot
        // products in FP32. KV's row-major layout is already K^T col-major.
        if (warp < kRows / 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> scores;
            wmma::fill_fragment(scores, 0.0f);
#pragma unroll
            for (int d = 0; d < kDim; d += 16) {
                wmma::load_matrix_sync(a, tile.q + d, kStride);
                wmma::load_matrix_sync(b, tile.kv + warp * 16 * kStride + d, kStride);
                wmma::mma_sync(scores, a, b, scores);
            }
            wmma::store_matrix_sync(tile.scores + warp * 16, scores, kRows, wmma::mem_row_major);
        }
        __syncthreads();

        // A 16-thread subgroup owns one head, with four scores per thread.
        // The denominator uses unrounded probabilities, as in the reference.
        const int h = threadIdx.x / 16;
        const int sublane = threadIdx.x % 16;
        float s[4];
        float maximum = tile.maximum[h];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = sublane + i * 16;
            const bool valid = first + r < n_idx && indices[first + r] >= 0;
            s[i] = valid ? tile.scores[h * kRows + r] * scale : -CUDART_INF_F;
            maximum = fmaxf(maximum, s[i]);
        }
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset, 16));
        const float alpha = expf(tile.maximum[h] - maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float p = s[i] == -CUDART_INF_F ? 0.0f : expf(s[i] - maximum);
            sum += p;
            tile.p[h * kProbStride + sublane + i * 16] = __float2bfloat16_rn(p);
        }
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, offset, 16);
        if (sublane == 0) {
            tile.maximum[h] = maximum;
            tile.sum[h] = tile.sum[h] * alpha + sum;
            tile.rescale[h] = alpha;
        }
        __syncthreads();

        // Preserve FP32 PV accumulators while changing their normalization
        // from the previous running maximum to this tile's running maximum.
        if (first != 0) {
#pragma unroll
            for (int v = 0; v < kOutputsPerWarp; ++v) {
                wmma::store_matrix_sync(stage, out[v], 16, wmma::mem_row_major);
                __syncwarp();
                for (int i = lane; i < 16 * 16; i += 32) stage[i] *= tile.rescale[i / 16];
                __syncwarp();
                wmma::load_matrix_sync(out[v], stage, 16, wmma::mem_row_major);
                __syncwarp();
            }
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> p;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> value;
#pragma unroll
        for (int r = 0; r < kRows; r += 16) {
            wmma::load_matrix_sync(p, tile.p + r, kProbStride);
#pragma unroll
            for (int v = 0; v < kOutputsPerWarp; ++v) {
                const int d = output_dim + (v * kWarps + warp) * 16;
                wmma::load_matrix_sync(value, tile.kv + r * kStride + d, kStride);
                wmma::mma_sync(out[v], p, value, out[v]);
            }
        }
        // No thread may overwrite KV, P, or scores while another warp still
        // uses the current tile. Q remains resident for the entire call.
        __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < kOutputsPerWarp; ++v) {
        wmma::store_matrix_sync(stage, out[v], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 16 * 16; i += 32) {
            const int h = i / 16;
            const int d = output_dim + (v * kWarps + warp) * 16 + i % 16;
            const float denom = tile.sum[h] + expf(sink[head + h] - tile.maximum[h]);
            output[(static_cast<size_t>(query) * kHeads + head + h) * kDim + d] =
                __float2bfloat16_rn(stage[i] / denom);
        }
        __syncwarp();
    }
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "sparse_attn_decode: %s: %s\n", operation, cudaGetErrorString(error));
        std::abort();
    }
}
}  // namespace

void sparse_attn_decode(const __nv_bfloat16* q, const __nv_bfloat16* window, const __nv_bfloat16* comp,
                        const int32_t* idx, int m, int n_idx, const float* sink, float scale,
                        __nv_bfloat16* o, cudaStream_t stream) {
    if (m <= 0) return;
    check_cuda(cudaFuncSetAttribute(fused_online, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   sizeof(TileStorage)), "opt in shared memory");
    fused_online<<<dim3(kHeads / kHeadTile, kDim / kOutputDim, m), kThreads, sizeof(TileStorage), stream>>>(
        q, window, comp, idx, n_idx, sink, scale, o);
}
}  // namespace strata::ds41::kernels
