// K3-12: scratch-free, full-list tensor-core attention.
// Each CTA handles eight heads and 128 output dimensions. The complete FP32
// score list is retained before a single full-list softmax. KV is streamed in
// 64-row, 128-dimension slices, then only the output slice is reread for PV.
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
constexpr int kMaxRows = 1024;
constexpr int kRows = 64;
constexpr int kSlice = 128;
constexpr int kQueryStride = kDim + 8;
constexpr int kSliceStride = kSlice + 8;
constexpr int kProbStride = kMaxRows + 8;
constexpr int kWarps = 4;
constexpr int kThreads = 32 * kWarps;
constexpr int kFragments = kRows / (16 * kWarps);
constexpr int kOutputFragments = kSlice / (16 * kWarps);
constexpr unsigned kWarpMask = 0xffffffffu;

struct __align__(32) TileStorage {
    bf16 q[kHeadTile * kQueryStride];
    bf16 kv[kRows * kSliceStride];
    union {
        float scores[kHeadTile][kMaxRows];
        bf16 probabilities[kHeadTile * kProbStride];
    } softmax;
    int indices[kMaxRows];
    float maximum[kHeadTile];
    float sum[kHeadTile];
};
static_assert(sizeof(TileStorage) == 62656, "shared-memory layout changed");
static_assert(sizeof(TileStorage) <= 99 * 1024, "consumer shared-memory limit");
static_assert(kMaxRows % kRows == 0 && kFragments == 1, "complete QK tiles");

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

// Predicated zero-fill never forms an address from an empty index. A block
// barrier after wait_group makes all lanes' asynchronous copies visible.
__device__ __forceinline__ void gather_slice(TileStorage& tile, const bf16* window,
                                             const bf16* comp, int first, int dimension) {
#pragma unroll
    for (int i = threadIdx.x; i < kRows * kSlice / 8; i += kThreads) {
        const int r = i / (kSlice / 8);
        const int d = (i % (kSlice / 8)) * 8;
        const int j = tile.indices[first + r];
        const bf16* source = window;
        if (j >= 0) {
            source = j < kWindow ? window + static_cast<size_t>(j) * kDim
                                 : comp + static_cast<size_t>(j - kWindow) * kDim;
            source += dimension + d;
        }
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(shared_address(tile.kv + r * kSliceStride + d)),
                        "l"(source), "r"(j >= 0 ? 16 : 0) : "memory");
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
}

__global__ __launch_bounds__(kThreads) void attention_full_list(
        const bf16* __restrict__ q, const bf16* __restrict__ window,
        const bf16* __restrict__ comp, const int32_t* __restrict__ idx,
        int n_idx, const float* __restrict__ sink, float scale,
        bf16* __restrict__ output) {
    extern __shared__ __align__(32) unsigned char storage[];
    TileStorage& tile = *reinterpret_cast<TileStorage*>(storage);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int query = blockIdx.z;
    const int output_dimension = blockIdx.y * kSlice;
    const int head = blockIdx.x * kHeadTile;
    const int head0 = (lane & 3) * 2;
    const int head1 = head0 + 1;
    const int mma_row = lane >> 2;
    const int32_t* indices = idx + static_cast<size_t>(query) * n_idx;
    const bf16* queries = q + (static_cast<size_t>(query) * kHeads + head) * kDim;

    for (int i = threadIdx.x; i < kHeadTile * kDim / 8; i += kThreads) {
        const int h = i / (kDim / 8);
        const int d = (i % (kDim / 8)) * 8;
        *reinterpret_cast<uint4*>(tile.q + h * kQueryStride + d) =
            *reinterpret_cast<const uint4*>(queries + h * kDim + d);
    }
    for (int r = threadIdx.x; r < kMaxRows; r += kThreads)
        tile.indices[r] = r < n_idx ? indices[r] : -1;
    __syncthreads();

    for (int first = 0; first < n_idx; first += kRows) {
        float score[4] = {};
        // Every output partition uses the same QK accumulation order.
        for (int dimension = 0; dimension < kDim; dimension += kSlice) {
            gather_slice(tile, window, comp, first, dimension);
#pragma unroll
            for (int d = 0; d < kSlice; d += 16) {
                unsigned query_fragment[2];
                load_b(query_fragment, tile.q + (lane % 8) * kQueryStride
                                                  + dimension + d + ((lane / 8) & 1) * 8);
                unsigned keys[4];
                load_a(keys, tile.kv + (warp * 16 + (lane % 16)) * kSliceStride
                                       + d + (lane / 16) * 8);
                mma(score, keys, query_fragment);
            }
            __syncthreads();  // all KV readers finish before the next gather
        }
        const int row = first + warp * 16 + mma_row;
        tile.softmax.scores[head0][row] = score[0];
        tile.softmax.scores[head1][row] = score[1];
        tile.softmax.scores[head0][row + 8] = score[2];
        tile.softmax.scores[head1][row + 8] = score[3];
    }
    __syncthreads();  // publish the entire list before the full-list reduction

    {
        // Sixteen lanes own each head. All 64 possible scores per lane must
        // be in registers before ANY compact BF16 store aliases the union.
        const int h = threadIdx.x / 16;
        const int sublane = threadIdx.x % 16;
        float scores[kMaxRows / 16];
        float maximum = -1.0e30f;
#pragma unroll
        for (int e = 0; e < kMaxRows / 16; ++e) {
            const int r = sublane + e * 16;
            scores[e] = tile.indices[r] >= 0 ? tile.softmax.scores[h][r] * scale : -CUDART_INF_F;
            maximum = fmaxf(maximum, scores[e]);
        }
#pragma unroll
        for (int offset = 8; offset; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(kWarpMask, maximum, offset, 16));
        __syncthreads();  // scores -> probabilities alias ownership transfer
        float sum = 0.0f;
#pragma unroll
        for (int e = 0; e < kMaxRows / 16; ++e) {
            const float p = scores[e] == -CUDART_INF_F ? 0.0f : expf(scores[e] - maximum);
            sum += p;  // denominator uses unrounded probabilities
            tile.softmax.probabilities[h * kProbStride + sublane + e * 16] = __float2bfloat16_rn(p);
        }
#pragma unroll
        for (int offset = 8; offset; offset >>= 1)
            sum += __shfl_xor_sync(kWarpMask, sum, offset, 16);
        if (sublane == 0) {
            tile.maximum[h] = maximum;
            tile.sum[h] = sum;
        }
    }
    __syncthreads();  // publish BF16 P and denominator before PV

    float result[kOutputFragments][4] = {};
    for (int first = 0; first < n_idx; first += kRows) {
        gather_slice(tile, window, comp, first, output_dimension);
#pragma unroll
        for (int r = 0; r < kRows; r += 16) {
            unsigned probabilities[2];
            load_b(probabilities, tile.softmax.probabilities + (lane % 8) * kProbStride
                                                                   + first + r + ((lane / 8) & 1) * 8);
#pragma unroll
            for (int v = 0; v < kOutputFragments; ++v) {
                const int d = (warp * kOutputFragments + v) * 16;
                unsigned values[4];
                load_a_transposed(values, tile.kv + (r + (lane % 8) + (lane / 16) * 8) * kSliceStride
                                                     + d + ((lane / 8) & 1) * 8);
                mma(result[v], values, probabilities);
            }
        }
        __syncthreads();  // all PV readers finish before the next gather
    }

    // The sink contributes only to the final unrounded denominator, never
    // the maximum or numerator. Empty lists match the reference's zero/inf.
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

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "sparse_attn_decode: %s: %s\n", operation, cudaGetErrorString(error));
        std::abort();
    }
}

void configure_shared_memory(cudaStream_t stream) {
    // No heap/device allocation, shared host mutation, or device sync. A
    // thread caches only its last device; switching devices rechecks the
    // function attribute. Concurrent eager callers set the identical value.
    static thread_local int configured_device = -1;
    int device = -1;
    check_cuda(cudaGetDevice(&device), "get current device");
    if (configured_device == device) return;
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes, attention_full_list), "get kernel attributes");
    if (attributes.maxDynamicSharedSizeBytes < static_cast<int>(sizeof(TileStorage))) {
        cudaStreamCaptureStatus status;
        check_cuda(cudaStreamIsCapturing(stream, &status), "check first-use capture status");
        if (status != cudaStreamCaptureStatusNone) {
            std::fprintf(stderr, "sparse_attn_decode: make an eager call on this device before graph capture\n");
            std::abort();
        }
        check_cuda(cudaFuncSetAttribute(attention_full_list, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       sizeof(TileStorage)), "opt in shared memory");
    }
    configured_device = device;
}
}  // namespace

void sparse_attn_decode(const bf16* q, const bf16* window, const bf16* comp,
                        const int32_t* idx, int m, int n_idx, const float* sink, float scale,
                        bf16* o, cudaStream_t stream) {
    if (m <= 0) return;
    if (m > 8 || n_idx < 0 || n_idx > kMaxRows) {
        std::fprintf(stderr, "sparse_attn_decode: invalid shape m=%d n_idx=%d\n", m, n_idx);
        std::abort();
    }
    configure_shared_memory(stream);
    attention_full_list<<<dim3(kHeads / kHeadTile, kDim / kSlice, m), kThreads, sizeof(TileStorage), stream>>>(
        q, window, comp, idx, n_idx, sink, scale, o);
}
}  // namespace strata::ds41::kernels
