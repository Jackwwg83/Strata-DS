// K3-09: direct-load SIMT attention. One CTA owns one (query, head), with
// register-resident Q and no tiled KV staging or global temporary storage.
#include "strata/ds41/kernels/k3_sparse_attn.hpp"

#include "strata/ds41/config.hpp"

#include <math_constants.h>
#include <cstdio>
#include <cstdlib>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
constexpr unsigned kFullWarp = 0xffffffffu;
constexpr int kWarps = 8;
constexpr int kThreads = 32 * kWarps;
constexpr int kMaxRows = 1024;
constexpr int kVector = 8;  // one aligned 128-bit load per lane
constexpr int kVectorsPerLane = kHeadDim / (32 * kVector);

struct __align__(16) SharedStorage {
    float probability[kMaxRows];
    // Reused between score normalization and the row-partitioned PV reduction.
    float partial[kWarps][kHeadDim];
    float maximum;
    float denominator;
};
static_assert(kHeadDim == 512 && kHeads == 64, "fixed K3 interface");
static_assert(sizeof(SharedStorage) == 20496, "shared storage layout");
static_assert(sizeof(SharedStorage) <= 48 * 1024, "no shared-memory opt-in");

__device__ __forceinline__ float2 unpack(unsigned int value) {
    // BF16-to-FP32 is an exact bit expansion. Keeping the packed loads explicit
    // avoids the baseline's sixteen scalar half-word loads per row and lane.
    return make_float2(__uint_as_float(value << 16), __uint_as_float(value & 0xffff0000u));
}

__device__ __forceinline__ const bf16* kv_row(const bf16* window, const bf16* comp, int j) {
    return j < kWindow ? window + static_cast<size_t>(j) * kHeadDim
                       : comp + static_cast<size_t>(j - kWindow) * kHeadDim;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(kFullWarp, value, offset);
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(kFullWarp, value, offset));
    return value;
}

__global__ __launch_bounds__(kThreads) void register_query_attention(
        const bf16* __restrict__ q, const bf16* __restrict__ window,
        const bf16* __restrict__ comp, const int32_t* __restrict__ idx,
        int n_idx, const float* __restrict__ sink, float scale,
        bf16* __restrict__ output) {
    __shared__ SharedStorage shared;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int head = blockIdx.x;
    const int query = blockIdx.y;
    const size_t query_head = static_cast<size_t>(query) * kHeads + head;
    const int32_t* indices = idx + static_cast<size_t>(query) * n_idx;

    // Every warp keeps its copy of Q in registers while scoring disjoint rows.
    // A warp loads two contiguous 512-byte spans per selected KV row.
    float query_values[kVectorsPerLane][kVector];
#pragma unroll
    for (int v = 0; v < kVectorsPerLane; ++v) {
        const int d = v * 32 * kVector + lane * kVector;
        const uint4 packed = *reinterpret_cast<const uint4*>(q + query_head * kHeadDim + d);
        const float2 a = unpack(packed.x), b = unpack(packed.y);
        const float2 c = unpack(packed.z), e = unpack(packed.w);
        query_values[v][0] = a.x; query_values[v][1] = a.y;
        query_values[v][2] = b.x; query_values[v][3] = b.y;
        query_values[v][4] = c.x; query_values[v][5] = c.y;
        query_values[v][6] = e.x; query_values[v][7] = e.y;
    }
    for (int row = warp; row < n_idx; row += kWarps) {
        const int j = indices[row];
        float score = 0.0f;
        if (j >= 0) {
            const bf16* source = kv_row(window, comp, j);
#pragma unroll
            for (int v = 0; v < kVectorsPerLane; ++v) {
                const int d = v * 32 * kVector + lane * kVector;
                const uint4 packed = *reinterpret_cast<const uint4*>(source + d);
                const float2 a = unpack(packed.x), b = unpack(packed.y);
                const float2 c = unpack(packed.z), e = unpack(packed.w);
                score += query_values[v][0] * a.x;
                score += query_values[v][1] * a.y;
                score += query_values[v][2] * b.x;
                score += query_values[v][3] * b.y;
                score += query_values[v][4] * c.x;
                score += query_values[v][5] * c.y;
                score += query_values[v][6] * e.x;
                score += query_values[v][7] * e.y;
            }
            score = warp_sum(score);
        }
        if (lane == 0) shared.probability[row] = j >= 0 ? score * scale : -CUDART_INF_F;
    }
    __syncthreads();

    // Complete-list normalization, including the -1e30 floor, reproduces the
    // reference's BF16 P definition instead of rounding against running maxima.
    float maximum = -1.0e30f;
    for (int row = tid; row < n_idx; row += kThreads)
        maximum = fmaxf(maximum, shared.probability[row]);
    maximum = warp_max(maximum);
    if (lane == 0) shared.partial[0][warp] = maximum;
    __syncthreads();
    if (tid == 0) {
        float result = -1.0e30f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) result = fmaxf(result, shared.partial[0][w]);
        shared.maximum = result;
    }
    __syncthreads();
    maximum = shared.maximum;
    float total = 0.0f;
    for (int row = tid; row < n_idx; row += kThreads) {
        const float score = shared.probability[row];
        const float p = score == -CUDART_INF_F ? 0.0f : expf(score - maximum);
        total += p;
        shared.probability[row] = __bfloat162float(__float2bfloat16_rn(p));
    }
    total = warp_sum(total);
    if (lane == 0) shared.partial[0][warp] = total;
    __syncthreads();
    if (tid == 0) {
        float result = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) result += shared.partial[0][w];
        // Sink participates only here, once, using the KV-only maximum.
        shared.denominator = result + expf(sink[head] - maximum);
    }
    __syncthreads();

    // Row partitioning cuts the baseline's serial PV chain by eight. Each
    // warp accumulates every output dimension for its own rows in registers;
    // one shared-memory reduction combines them without atomics or scratch.
    float value[kVectorsPerLane][kVector] = {};
    for (int row = warp; row < n_idx; row += kWarps) {
        const int j = indices[row];
        if (j < 0) continue;
        const bf16* source = kv_row(window, comp, j);
        const float p = shared.probability[row];
#pragma unroll
        for (int v = 0; v < kVectorsPerLane; ++v) {
            const int d = v * 32 * kVector + lane * kVector;
            const uint4 packed = *reinterpret_cast<const uint4*>(source + d);
            const float2 a = unpack(packed.x), b = unpack(packed.y);
            const float2 c = unpack(packed.z), e = unpack(packed.w);
            value[v][0] += p * a.x; value[v][1] += p * a.y;
            value[v][2] += p * b.x; value[v][3] += p * b.y;
            value[v][4] += p * c.x; value[v][5] += p * c.y;
            value[v][6] += p * e.x; value[v][7] += p * e.y;
        }
    }
#pragma unroll
    for (int v = 0; v < kVectorsPerLane; ++v) {
        const int d = v * 32 * kVector + lane * kVector;
        *reinterpret_cast<float4*>(&shared.partial[warp][d]) =
            make_float4(value[v][0], value[v][1], value[v][2], value[v][3]);
        *reinterpret_cast<float4*>(&shared.partial[warp][d + 4]) =
            make_float4(value[v][4], value[v][5], value[v][6], value[v][7]);
    }
    __syncthreads();
    float2 result = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        const float2 partial = *reinterpret_cast<const float2*>(&shared.partial[w][tid * 2]);
        result.x += partial.x;
        result.y += partial.y;
    }
    const float denominator = shared.denominator;
    *reinterpret_cast<__nv_bfloat162*>(output + query_head * kHeadDim + tid * 2) =
        __floats2bfloat162_rn(result.x / denominator, result.y / denominator);
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
    // Identical eager/capture execution: a single launch on the caller's stream.
    register_query_attention<<<dim3(kHeads, m), kThreads, 0, stream>>>(
        q, window, comp, idx, n_idx, sink, scale, o);
}
}  // namespace strata::ds41::kernels
