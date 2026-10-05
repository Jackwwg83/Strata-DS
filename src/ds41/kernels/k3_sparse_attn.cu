// BF16 tensor-core sparse attention. Separate QK, softmax, and PV stages keep
// the reference's full-list maximum and its single BF16 rounding of P.
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
constexpr int kHeadTile = 16;
constexpr int kRowTile = 32;
constexpr int kDotTile = 128;
constexpr int kDotStride = kDotTile + 8;
constexpr int kValueTile = 32;
constexpr int kProbStride = kRowTile + 8;
constexpr int kValueStride = kValueTile + 8;
constexpr int kThreads = 64;

__device__ __forceinline__ const bf16* kv_row(const bf16* window, const bf16* comp, int j) {
    return j < kWindow ? window + static_cast<size_t>(j) * kDim
                       : comp + static_cast<size_t>(j - kWindow) * kDim;
}

// One CTA computes 16 heads against 32 selected rows. Each KV row is loaded
// once and reused across every head; the two warps own disjoint 16-row halves.
// A 128-wide, padded dimension tile avoids shared-memory bank conflicts and
// limits static shared memory to 12.75 KiB without an opt-in attribute.
__global__ void qk_tiles(const bf16* __restrict__ q, const bf16* __restrict__ window,
                         const bf16* __restrict__ comp, const int32_t* __restrict__ idx,
                         int n_idx, int stride, float scale, float* __restrict__ scores) {
    __shared__ __align__(32) bf16 qs[kHeadTile * kDotStride];
    __shared__ __align__(32) bf16 kvs[kRowTile * kDotStride];
    const int query = blockIdx.z;
    const int head = blockIdx.y * kHeadTile;
    const int row = blockIdx.x * kRowTile;
    const int warp = threadIdx.x >> 5;
    const bf16* qbase = q + (static_cast<size_t>(query) * kHeads + head) * kDim;
    const int32_t* indices = idx + static_cast<size_t>(query) * n_idx;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int tile = 0; tile < kDim; tile += kDotTile) {
        for (int i = threadIdx.x; i < kHeadTile * kDotTile / 8; i += kThreads) {
            const int h = i / (kDotTile / 8);
            const int d = (i % (kDotTile / 8)) * 8;
            *reinterpret_cast<uint4*>(qs + h * kDotStride + d) =
                *reinterpret_cast<const uint4*>(qbase + h * kDim + tile + d);
        }
        for (int i = threadIdx.x; i < kRowTile * kDotTile / 8; i += kThreads) {
            const int r = i / (kDotTile / 8);
            const int d = (i % (kDotTile / 8)) * 8;
            const int j = row + r < n_idx ? indices[row + r] : -1;
            const uint4 v = j >= 0 ? *reinterpret_cast<const uint4*>(kv_row(window, comp, j) + tile + d)
                                  : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(kvs + r * kDotStride + d) = v;
        }
        __syncthreads();
#pragma unroll
        for (int d = 0; d < kDotTile; d += 16) {
            wmma::load_matrix_sync(a, qs + d, kDotStride);
            // KV is row-major [position, dimension], so its transpose is
            // already a column-major operand without a separate copy.
            wmma::load_matrix_sync(b, kvs + warp * 16 * kDotStride + d, kDotStride);
            wmma::mma_sync(c, a, b, c);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < c.num_elements; ++i) c.x[i] *= scale;
    float* out = scores + (static_cast<size_t>(query) * kHeads + head) * stride + row;
    wmma::store_matrix_sync(out + warp * 16, c, stride, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < kHeadTile * kRowTile; i += kThreads) {
        const int r = i % kRowTile;
        float* s = out + (i / kRowTile) * stride + r;
        const bool valid = row + r < n_idx && indices[row + r] >= 0;
        if (!valid) *s = -CUDART_INF_F;
    }
}

template <bool Maximum>
__device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
    for (int offset = 16; offset != 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, v, offset);
        v = Maximum ? fmaxf(v, other) : v + other;
    }
    return v;
}

// The FP32 score row becomes a BF16 probability row in place. All scores are
// first retained in registers, so compacting the row cannot overwrite input
// that another thread still needs. Different heads have disjoint allocations.
__global__ void softmax_rows(float* scores, int stride, const float* __restrict__ sink,
                             float* __restrict__ denominators) {
    __shared__ float reductions[16];
    const int h = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float* row = scores + static_cast<size_t>(h) * stride;
    float values[4];
    float mx = -1.0e30f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = threadIdx.x + i * 256;
        values[i] = col < stride ? row[col] : -CUDART_INF_F;
        mx = fmaxf(mx, values[i]);
    }
    mx = warp_reduce<true>(mx);
    if (lane == 0) reductions[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        mx = lane < 8 ? reductions[lane] : -1.0e30f;
        mx = warp_reduce<true>(mx);
        if (lane == 0) reductions[0] = mx;
    }
    __syncthreads();
    mx = reductions[0];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i] = values[i] == -CUDART_INF_F ? 0.0f : expf(values[i] - mx);
        sum += values[i];
    }
    sum = warp_reduce<false>(sum);
    // A separate half of reductions keeps the preceding maximum readable by
    // every warp until each has consumed it.
    if (lane == 0) reductions[8 + warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? reductions[8 + lane] : 0.0f;
        sum = warp_reduce<false>(sum);
        if (lane == 0) denominators[h] = sum + expf(sink[h % kHeads] - mx);
    }
    bf16* probabilities = reinterpret_cast<bf16*>(row);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = threadIdx.x + i * 256;
        if (col < stride) probabilities[col] = __float2bfloat16_rn(values[i]);
    }
}

// One CTA owns [16 heads, 32 output dimensions]. Its two warps keep the FP32
// PV accumulators in registers, walking over 32-row KV tiles shared by all
// 16 heads. BF16 P is multiplied without an extra rounding or normalization.
__global__ void pv_tiles(const float* scores, const float* __restrict__ denominators,
                         const bf16* __restrict__ window, const bf16* __restrict__ comp,
                         const int32_t* __restrict__ idx, int n_idx, int stride,
                         bf16* __restrict__ o) {
    __shared__ __align__(32) bf16 ps[kHeadTile * kProbStride];
    __shared__ __align__(32) bf16 vs[kRowTile * kValueStride];
    __shared__ __align__(32) float result[kHeadTile * kValueTile];
    const int query = blockIdx.z;
    const int head = blockIdx.y * kHeadTile;
    const int dim = blockIdx.x * kValueTile;
    const int warp = threadIdx.x >> 5;
    const int32_t* indices = idx + static_cast<size_t>(query) * n_idx;
    const float* sbase = scores + (static_cast<size_t>(query) * kHeads + head) * stride;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int row = 0; row < stride; row += kRowTile) {
        for (int i = threadIdx.x; i < kHeadTile * kRowTile / 8; i += kThreads) {
            const int h = i / (kRowTile / 8);
            const int p = (i % (kRowTile / 8)) * 8;
            const bf16* src = reinterpret_cast<const bf16*>(sbase + h * stride) + row + p;
            *reinterpret_cast<uint4*>(ps + h * kProbStride + p) = *reinterpret_cast<const uint4*>(src);
        }
        for (int i = threadIdx.x; i < kRowTile * kValueTile / 8; i += kThreads) {
            const int r = i / (kValueTile / 8);
            const int d = (i % (kValueTile / 8)) * 8;
            const int j = row + r < n_idx ? indices[row + r] : -1;
            const uint4 v = j >= 0 ? *reinterpret_cast<const uint4*>(kv_row(window, comp, j) + dim + d)
                                  : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(vs + r * kValueStride + d) = v;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kRowTile; k += 16) {
            wmma::load_matrix_sync(a, ps + k, kProbStride);
            wmma::load_matrix_sync(b, vs + k * kValueStride + warp * 16, kValueStride);
            wmma::mma_sync(c, a, b, c);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(result + warp * 16, c, kValueTile, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < kHeadTile * kValueTile; i += kThreads) {
        const int h = query * kHeads + head + i / kValueTile;
        const int d = dim + i % kValueTile;
        o[static_cast<size_t>(h) * kDim + d] = __float2bfloat16_rn(result[i] / denominators[h]);
    }
}

void check_cuda(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "sparse_attn_decode: %s: %s\n", where, cudaGetErrorString(error));
        std::abort();
    }
}

}  // namespace

void sparse_attn_decode(const __nv_bfloat16* q, const __nv_bfloat16* window, const __nv_bfloat16* comp,
                        const int32_t* idx, int m, int n_idx, const float* sink, float scale,
                        __nv_bfloat16* o, cudaStream_t stream) {
    if (m <= 0) return;
    // A private stream-ordered allocation avoids global caches and works for
    // overlapping calls on different streams. CUDA 12.8 supports the pool API
    // on every required target; freeing is ordered after the last consumer.
    const int stride = n_idx > 0 ? (n_idx + kRowTile - 1) / kRowTile * kRowTile : kRowTile;
    const size_t heads = static_cast<size_t>(m) * kHeads;
    float* scratch = nullptr;
    check_cuda(cudaMallocAsync(reinterpret_cast<void**>(&scratch),
                               (heads * stride + heads) * sizeof(float), stream), "allocate workspace");
    float* denominators = scratch + heads * stride;
    qk_tiles<<<dim3(stride / kRowTile, kHeads / kHeadTile, m), kThreads, 0, stream>>>(
        q, window, comp, idx, n_idx, stride, scale, scratch);
    softmax_rows<<<static_cast<unsigned>(heads), 256, 0, stream>>>(scratch, stride, sink, denominators);
    pv_tiles<<<dim3(kDim / kValueTile, kHeads / kHeadTile, m), kThreads, 0, stream>>>(
        scratch, denominators, window, comp, idx, n_idx, stride, o);
    check_cuda(cudaFreeAsync(scratch, stream), "release workspace");
}

}  // namespace strata::ds41::kernels
