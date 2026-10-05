// Split-KV sparse attention: independent local softmax partitions, followed by
// a stable merge. All query windows share the two launches on the caller's stream.
#include "strata/ds41/kernels/k3_sparse_attn.hpp"

#include "strata/ds41/config.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::ds41::kernels {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPartition = 128;
constexpr int kMaxPartitions = (1024 + kPartition - 1) / kPartition;
constexpr unsigned kFullWarp = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset; offset >>= 1)
        v += __shfl_down_sync(kFullWarp, v, offset);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(kFullWarp, v, offset));
    return v;
}

template <bool Maximum>
__device__ __forceinline__ float block_reduce(float v, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = Maximum ? warp_max(v) : warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < kWarps ? scratch[lane] : (Maximum ? -1e30f : 0.0f);
        v = Maximum ? warp_max(v) : warp_sum(v);
        if (lane == 0) scratch[0] = v;
    }
    __syncthreads();
    const float result = scratch[0];
    // Every thread must consume the result before the next reduction can reuse
    // scratch[0]. This also publishes all probability stores before the PV loop.
    __syncthreads();
    return result;
}

// One block computes one (query, head, key partition). The dot-product order
// matches the reference's lane-strided FP32 sum. Each partition rounds its P
// relative to its own maximum, as allowed for online softmax by the task spec.
__global__ void sparse_partials(const bf16* __restrict__ q,
                                const bf16* __restrict__ window,
                                const bf16* __restrict__ comp,
                                const int32_t* __restrict__ idx,
                                int n_idx, int partitions, float scale,
                                float* __restrict__ numerator,
                                float2* __restrict__ stats) {
    __shared__ const bf16* rows[kPartition];
    __shared__ float probability[kPartition];
    __shared__ float reduction[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int head = blockIdx.x;
    const int part = blockIdx.y;
    const int query = blockIdx.z;
    const int query_head = query * kHeads + head;
    const int begin = part * kPartition;
    const int count = min(kPartition, max(0, n_idx - begin));
    const int out_part = query_head * partitions + part;

    if (tid < kPartition) {
        const int j = tid < count ? idx[(size_t) query * n_idx + begin + tid] : -1;
        rows[tid] = j < 0 ? nullptr
            : (j < kWindow ? window + (size_t) j * kHeadDim
                           : comp + (size_t) (j - kWindow) * kHeadDim);
    }

    float q_lane[kHeadDim / 32];
#pragma unroll
    for (int d = 0; d < kHeadDim / 32; ++d)
        q_lane[d] = __bfloat162float(q[(size_t) query_head * kHeadDim + d * 32 + lane]);
    __syncthreads();

    for (int key = warp; key < kPartition; key += kWarps) {
        const bf16* row = rows[key];
        float score = 0.0f;
        if (row) {
#pragma unroll
            for (int d = 0; d < kHeadDim / 32; ++d)
                score += q_lane[d] * __bfloat162float(row[d * 32 + lane]);
            score = warp_sum(score);
        }
        if (lane == 0) probability[key] = row ? score * scale : -INFINITY;
    }
    __syncthreads();

    float mx = tid < kPartition ? probability[tid] : -INFINITY;
    mx = fmaxf(block_reduce<true>(mx, reduction), -1e30f);
    float p = 0.0f;
    if (tid < kPartition) {
        const float score = probability[tid];
        p = score == -INFINITY ? 0.0f : expf(score - mx);
        probability[tid] = __bfloat162float(__float2bfloat16_rn(p));
    }
    const float denominator = block_reduce<false>(p, reduction);
    if (tid == 0) stats[out_part] = make_float2(mx, denominator);

    // Two coalesced dimension strips per thread, independent FP32 accumulators.
    float a0 = 0.0f;
    float a1 = 0.0f;
#pragma unroll 4
    for (int key = 0; key < count; ++key) {
        const bf16* row = rows[key];
        if (row) {
            const float weight = probability[key];
            a0 += weight * __bfloat162float(row[tid]);
            a1 += weight * __bfloat162float(row[tid + kThreads]);
        }
    }
    float* out = numerator + (size_t) out_part * kHeadDim;
    out[tid] = a0;
    out[tid + kThreads] = a1;
}

// Merge unnormalized partial numerators with the same exponential rescaling as
// an online softmax. The sink contributes once, only to the final denominator.
__global__ void sparse_merge(const float* __restrict__ numerator,
                             const float2* __restrict__ stats,
                             int partitions, const float* __restrict__ sink,
                             bf16* __restrict__ o) {
    __shared__ float factor[kMaxPartitions];
    __shared__ float denominator;
    const int query_head = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid == 0) {
        float mx = -1e30f;
        for (int part = 0; part < partitions; ++part)
            mx = fmaxf(mx, stats[query_head * partitions + part].x);
        float sum = 0.0f;
        for (int part = 0; part < partitions; ++part) {
            const float2 local = stats[query_head * partitions + part];
            const float rescale = expf(local.x - mx);
            factor[part] = rescale;
            sum += local.y * rescale;
        }
        denominator = sum + expf(sink[query_head % kHeads] - mx);
    }
    __syncthreads();
    float a0 = 0.0f;
    float a1 = 0.0f;
    for (int part = 0; part < partitions; ++part) {
        const float* in = numerator + ((size_t) query_head * partitions + part) * kHeadDim;
        a0 += factor[part] * in[tid];
        a1 += factor[part] * in[tid + kThreads];
    }
    bf16* out = o + (size_t) query_head * kHeadDim;
    out[tid] = __float2bfloat16_rn(a0 / denominator);
    out[tid + kThreads] = __float2bfloat16_rn(a1 / denominator);
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "sparse_attn_decode: %s: %s\n", operation, cudaGetErrorString(status));
        std::abort();
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
    const int partitions = n_idx > 0 ? (n_idx + kPartition - 1) / kPartition : 1;
    const size_t parts = (size_t) m * kHeads * partitions;
    const size_t numerator_bytes = parts * kHeadDim * sizeof(float);
    float* workspace = nullptr;
    // Stream-ordered scratch is private to this call. There is no shared mutable
    // workspace, no default-stream assumption, and no device synchronization.
    check_cuda(cudaMallocAsync(reinterpret_cast<void**>(&workspace),
                               numerator_bytes + parts * sizeof(float2), stream), "allocate partials");
    auto* stats = reinterpret_cast<float2*>(reinterpret_cast<char*>(workspace) + numerator_bytes);
    sparse_partials<<<dim3(kHeads, partitions, m), kThreads, 0, stream>>>(
        q, window, comp, idx, n_idx, partitions, scale, workspace, stats);
    check_cuda(cudaGetLastError(), "partial kernel");
    sparse_merge<<<m * kHeads, kThreads, 0, stream>>>(workspace, stats, partitions, sink, o);
    check_cuda(cudaGetLastError(), "merge kernel");
    check_cuda(cudaFreeAsync(workspace, stream), "release partials");
}

}  // namespace strata::ds41::kernels
