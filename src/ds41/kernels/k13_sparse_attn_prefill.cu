// K13-01: one fused CTA per query and sixteen heads, with BF16 tensor-core
// QK and PV. Keep every logit in shared memory; no global score scratch.
#include "strata/ds41/kernels/k13_sparse_attn_prefill.hpp"

#include <math_constants.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_set>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
constexpr int kHeads = 64;
constexpr int kDim = 512;
constexpr int kHeadTile = 16;
constexpr int kRows = 64;
constexpr int kSlice = 128;
constexpr int kMaxIndices = 1024;
constexpr int kQueryStride = kDim + 8;
constexpr int kProbStride = kMaxIndices + 8;
constexpr int kThreads = 256;
constexpr int kOutputFragments = 8;
constexpr unsigned kWarpMask = 0xffffffffu;

struct __align__(32) TileStorage {
    union {
        struct {
            bf16 q[kHeadTile * kQueryStride];
            float scores[kHeadTile][kMaxIndices];
            bf16 kv[kRows * kSlice];
        } qk;
        struct {
            bf16 probabilities[kHeadTile * kProbStride];
            bf16 kv[kRows * kDim];
        } pv;
    } phase;
    const bf16* sources[kRows];
    int valid[kRows];
    float denominator[kHeadTile];
};
static_assert(sizeof(TileStorage) == 99392, "shared-memory layout changed");
static_assert(sizeof(TileStorage) <= 99 * 1024, "consumer GPU shared-memory limit");

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

// Swizzle complete aligned 16-byte vectors, preserving each eight-element
// row segment expected by ldmatrix while spreading adjacent rows over banks.
template <int Width>
__device__ __forceinline__ int kv_offset(int row, int dimension) {
    return row * Width + (dimension ^ ((row & 7) * 8));
}

__device__ __forceinline__ void select_rows(TileStorage& tile, const bf16* kv,
                                          const int32_t* indices, int first, int n_idx) {
    if (threadIdx.x < kRows) {
        const int position = first + threadIdx.x;
        const int j = position < n_idx ? indices[position] : -1;
        // Cast BEFORE multiplication. Every listed nonnegative row is read
        // directly from kv, without assumptions about order or window layout.
        tile.sources[threadIdx.x] = j >= 0 ? kv + static_cast<size_t>(j) * kDim : kv;
        tile.valid[threadIdx.x] = j >= 0;
    }
    __syncthreads();
}

template <int Width>
__device__ __forceinline__ void gather(TileStorage& tile, bf16* destination, int dimension) {
#pragma unroll
    for (int i = threadIdx.x; i < kRows * Width / 8; i += kThreads) {
        const int r = i / (Width / 8);
        const int d = (i % (Width / 8)) * 8;
        const int valid = tile.valid[r];
        const bf16* source = tile.sources[r];
        if (valid) source += dimension + d;
        // A negative/padded row copies zero bytes from the safe base pointer.
        // Never form an invalid global address, including for the final tile.
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(shared_address(destination + kv_offset<Width>(r, d))),
                        "l"(source), "r"(valid ? 16 : 0) : "memory");
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
}

__global__ __launch_bounds__(kThreads) void attention_fused(
        const bf16* __restrict__ q, const bf16* __restrict__ kv,
        const int32_t* __restrict__ idx, int n_idx, const float* __restrict__ sink,
        float scale, bf16* __restrict__ output) {
    extern __shared__ __align__(32) unsigned char storage[];
    TileStorage& tile = *reinterpret_cast<TileStorage*>(storage);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int query = blockIdx.y;
    const int head = blockIdx.x * kHeadTile;
    const int head_group = (warp >> 2) * 8;
    const int head0 = head_group + (lane & 3) * 2;
    const int head1 = head0 + 1;
    const int mma_row = lane >> 2;
    const int key_tile = (warp & 3) * 16;
    const int32_t* indices = idx + static_cast<size_t>(query) * n_idx;
    const bf16* queries = q + (static_cast<size_t>(query) * kHeads + head) * kDim;

    for (int i = threadIdx.x; i < kHeadTile * kDim / 8; i += kThreads) {
        const int h = i / (kDim / 8);
        const int d = (i % (kDim / 8)) * 8;
        *reinterpret_cast<uint4*>(tile.phase.qk.q + h * kQueryStride + d) =
            *reinterpret_cast<const uint4*>(queries + h * kDim + d);
    }
    __syncthreads();

    for (int first = 0; first < n_idx; first += kRows) {
        select_rows(tile, kv, indices, first, n_idx);
        float scores[4] = {};
        for (int dimension = 0; dimension < kDim; dimension += kSlice) {
            gather<kSlice>(tile, tile.phase.qk.kv, dimension);
#pragma unroll
            for (int d = 0; d < kSlice; d += 16) {
                unsigned keys[4], queries_fragment[2];
                load_a(keys, tile.phase.qk.kv + kv_offset<kSlice>(
                    key_tile + (lane % 16), d + (lane / 16) * 8));
                load_b(queries_fragment, tile.phase.qk.q +
                    (head_group + lane % 8) * kQueryStride + dimension + d + ((lane / 8) & 1) * 8);
                mma(scores, keys, queries_fragment);
            }
            __syncthreads();  // retire every reader before the next gather
        }
        const int row = key_tile + mma_row;
        tile.phase.qk.scores[head0][first + row] = tile.valid[row] ? scores[0] * scale : -CUDART_INF_F;
        tile.phase.qk.scores[head1][first + row] = tile.valid[row] ? scores[1] * scale : -CUDART_INF_F;
        tile.phase.qk.scores[head0][first + row + 8] = tile.valid[row + 8] ? scores[2] * scale : -CUDART_INF_F;
        tile.phase.qk.scores[head1][first + row + 8] = tile.valid[row + 8] ? scores[3] * scale : -CUDART_INF_F;
        __syncthreads();  // protect validity until every score is written
    }

    // Sixteen lanes reduce each head across the entire list, exactly as the
    // reference: max with a -1e30 floor, exp rounded to BF16 BEFORE PV, and
    // division by the unrounded denominator AFTER PV. Sink is denominator-only.
    // Hold all scores in registers before the union changes phase: even heads
    // whose probability rows overlap other score rows cannot overwrite readers.
    const int h = threadIdx.x / 16;
    const int sublane = threadIdx.x % 16;
    float probabilities[kMaxIndices / 16];
    float maximum = -1.0e30f;
#pragma unroll
    for (int e = 0; e < kMaxIndices / 16; ++e) {
        const int r = sublane + e * 16;
        probabilities[e] = r < n_idx ? tile.phase.qk.scores[h][r] : -CUDART_INF_F;
        maximum = fmaxf(maximum, probabilities[e]);
    }
#pragma unroll
    for (int offset = 8; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(kWarpMask, maximum, offset, 16));
    __syncthreads();  // all score reads complete before ANY union overwrite
    float sum = 0.0f;
#pragma unroll
    for (int e = 0; e < kMaxIndices / 16; ++e) {
        const float p = probabilities[e] == -CUDART_INF_F ? 0.0f : expf(probabilities[e] - maximum);
        sum += p;
        tile.phase.pv.probabilities[h * kProbStride + sublane + e * 16] = __float2bfloat16_rn(p);
    }
#pragma unroll
    for (int offset = 8; offset; offset >>= 1)
        sum += __shfl_xor_sync(kWarpMask, sum, offset, 16);
    if (sublane == 0) tile.denominator[h] = sum + expf(sink[head + h] - maximum);
    __syncthreads();  // publish probabilities, denominator, and phase transition

    float result[kOutputFragments][4] = {};
    for (int first = 0; first < n_idx; first += kRows) {
        select_rows(tile, kv, indices, first, n_idx);
        gather<kDim>(tile, tile.phase.pv.kv, 0);
#pragma unroll
        for (int r = 0; r < kRows; r += 16) {
            unsigned p[2];
            load_b(p, tile.phase.pv.probabilities + (head_group + lane % 8) * kProbStride
                         + first + r + ((lane / 8) & 1) * 8);
#pragma unroll
            for (int v = 0; v < kOutputFragments; ++v) {
                const int d = ((warp & 3) * kOutputFragments + v) * 16;
                unsigned values[4];
                load_a_transposed(values, tile.phase.pv.kv + kv_offset<kDim>(
                    r + (lane % 8) + (lane / 16) * 8, d + ((lane / 8) & 1) * 8));
                mma(result[v], values, p);
            }
        }
        __syncthreads();  // finish all PV readers before overwriting KV
    }

    const float denominator0 = tile.denominator[head0];
    const float denominator1 = tile.denominator[head1];
    bf16* out0 = output + (static_cast<size_t>(query) * kHeads + head + head0) * kDim;
    bf16* out1 = output + (static_cast<size_t>(query) * kHeads + head + head1) * kDim;
#pragma unroll
    for (int v = 0; v < kOutputFragments; ++v) {
        const int d = ((warp & 3) * kOutputFragments + v) * 16 + mma_row;
        out0[d] = __float2bfloat16_rn(result[v][0] / denominator0);
        out1[d] = __float2bfloat16_rn(result[v][1] / denominator1);
        out0[d + 8] = __float2bfloat16_rn(result[v][2] / denominator0);
        out1[d + 8] = __float2bfloat16_rn(result[v][3] / denominator1);
    }
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "sparse_attn_prefill: %s: %s\n", operation, cudaGetErrorString(status));
        std::abort();
    }
}

// Attributes belong to a device/function pair. Configure each device on its
// first eager call, before graph capture. No scratch allocation is needed.
void configure_shared_memory() {
    static std::mutex mutex;
    static std::unordered_set<int> initialized;
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized.find(device) == initialized.end()) {
        check_cuda(cudaFuncSetAttribute(attention_fused, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        sizeof(TileStorage)), "shared-memory opt-in");
        initialized.insert(device);
    }
}
}  // namespace

void sparse_attn_prefill(const bf16* q, const bf16* kv, const int32_t* idx, int m, int n_idx,
                         const float* sink, float scale, bf16* o, cudaStream_t stream) {
    if (m <= 0) return;
    if (m > 16384 || n_idx < 0 || n_idx > kMaxIndices) {
        std::fprintf(stderr, "sparse_attn_prefill: invalid shape m=%d n_idx=%d\n", m, n_idx);
        std::abort();
    }
    configure_shared_memory();
    attention_fused<<<dim3(kHeads / kHeadTile, m), kThreads, sizeof(TileStorage), stream>>>(
        q, kv, idx, n_idx, sink, scale, o);
}
}  // namespace strata::ds41::kernels
