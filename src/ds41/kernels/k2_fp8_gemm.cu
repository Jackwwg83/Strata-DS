// K2: decode weights once per invocation in bounded caller-workspace chunks.
// BF16 operands are shared by all M tiles, with a double-buffered cp.async GEMM.
// Small M uses a tiled dequant fallback to avoid many narrow decode launches.
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"

#include <cuda_fp8.h>

namespace strata::ds41::kernels {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kTileM = 64;
constexpr int kTileN = 256;
constexpr int kTileK = 32;
constexpr int kStride = kTileK + 8;
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;

// Match ops::act_quant_to_f32_k, including exact power-of-two boundaries.
__device__ __forceinline__ float round_pow2(float value) {
    int exponent;
    const float mantissa = frexpf(value, &exponent);
    return ldexpf(1.0f, mantissa == 0.5f ? exponent - 1 : exponent);
}

__global__ void quantize_activations(const bf16* x, bf16* quantized, int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // K % 32 == 0 makes this branch uniform within every warp, including M tails.
    if (index >= elements) return;
    const float value = __bfloat162float(x[index]);
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    const float scale = round_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    const __nv_fp8_e4m3 q(fminf(fmaxf(value / scale, -448.0f), 448.0f));
    quantized[index] = __float2bfloat16_rn(float(q) * scale);
}

struct __align__(32) Operands {
    bf16 a[kTileM * kStride];
    bf16 b[kTileN * kStride];
};
struct __align__(32) SharedStorage {
    Operands operands;
};
static_assert(sizeof(SharedStorage) == 25 * 1024, "Unexpected shared-memory layout");
static_assert(sizeof(SharedStorage) <= 99 * 1024, "K2 shared-memory limit exceeded");

struct RegisterStage {
    uint4 a;
    uint4 b;
    float scale;
};

// K is divisible by 32, so each 16-byte vector lies within an input row. Only
// half of the CTA loads A; every thread loads sixteen packed weights. One scale
// load per warp suffices because its sixteen adjacent columns share a 32x32 block.
__device__ __forceinline__ RegisterStage prefetch(
    const bf16* activation, const uint8_t* weight, const uint8_t* scales,
    int64_t row_base, int64_t col_base, int64_t k_base,
    int64_t M, int64_t N, int64_t K) {
    RegisterStage next;
    next.a = make_uint4(0, 0, 0, 0);
    next.b = make_uint4(0, 0, 0, 0);
    const int a_row = threadIdx.x / 4;
    const int a_k = (threadIdx.x % 4) * 8;
    const int64_t row = row_base + a_row;
    if (threadIdx.x < kTileM * 4 && row < M)
        next.a = *reinterpret_cast<const uint4*>(activation + row * K + k_base + a_k);
    const int b_col = threadIdx.x / 2;
    const int b_k = (threadIdx.x % 2) * 16;
    const int64_t col = col_base + b_col;
    if (col < N)
        next.b = *reinterpret_cast<const uint4*>(weight + col * K + k_base + b_k);
    float scale = 1.0f;
    if ((threadIdx.x % 32) == 0 && col < N)
        scale = ldexpf(1.0f, int(scales[(col / 32) * (K / 32) + k_base / 32]) - 127);
    next.scale = __shfl_sync(0xffffffffu, scale, 0);
    return next;
}

// Decode a packed group with the CUDA FP8 conversion on every supported target.
// Scaling is FP32 before the exact BF16 conversion, matching the reference even
// for FP8 subnormals, signed zero, and scales outside the benchmark's range.
__device__ __forceinline__ void store_four(bf16* dst, unsigned packed, float scale) {
    __nv_fp8x4_e4m3 q;
    q.__x = packed;
    const float4 value = static_cast<float4>(q);
    auto* pair = reinterpret_cast<__nv_bfloat162*>(dst);
    pair[0] = __floats2bfloat162_rn(value.x * scale, value.y * scale);
    pair[1] = __floats2bfloat162_rn(value.z * scale, value.w * scale);
}

__device__ __forceinline__ void publish(Operands& shared, const RegisterStage& next) {
    const int a_row = threadIdx.x / 4;
    const int a_k = (threadIdx.x % 4) * 8;
    if (threadIdx.x < kTileM * 4)
        *reinterpret_cast<uint4*>(shared.a + a_row * kStride + a_k) = next.a;
    const int b_col = threadIdx.x / 2;
    const int b_k = (threadIdx.x % 2) * 16;
    bf16* dst = shared.b + b_col * kStride + b_k;
    store_four(dst,      next.b.x, next.scale);
    store_four(dst + 4,  next.b.y, next.scale);
    store_four(dst + 8,  next.b.z, next.scale);
    store_four(dst + 12, next.b.w, next.scale);
}

// Operand register mappings follow PTX mma.m16n8k16: A is four packed BF16
// registers; B is two. Both shared arrays store contiguous K, so neither
// ldmatrix uses .trans. The B operand is logically column-major KxN.
__device__ __forceinline__ void load_a(unsigned (&a)[4], const bf16* ptr) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(address) : "memory");
}

__device__ __forceinline__ void load_b(unsigned (&b)[2], const bf16* ptr) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(address) : "memory");
}

__device__ __forceinline__ void mma(float (&d)[4], const unsigned (&a)[4],
                                    const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
}

// The 4x4 warp grid assigns a 16x64 output rectangle to each warp. Each lane
// holds 32 accumulator floats, half as many as a 32x64 warp tile. The next tile
// remains packed in registers while tensor cores consume the shared tile.
__global__ __launch_bounds__(kThreads, 2) void gemm_register_stage(
    const bf16* __restrict__ activation, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales, bf16* __restrict__ output,
    int64_t M, int64_t N, int64_t K) {
    __shared__ SharedStorage shared;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = (warp / 4) * 16;
    const int warp_n = (warp % 4) * 64;
    const int64_t row_base = static_cast<int64_t>(blockIdx.y) * kTileM;
    const int64_t col_base = static_cast<int64_t>(blockIdx.x) * kTileN;
    float accum[8][4] = {};

    if (K > 0) {
        const RegisterStage first = prefetch(activation, weight, scales,
                                            row_base, col_base, 0, M, N, K);
        publish(shared.operands, first);
        __syncthreads();
    }
    for (int64_t k_base = 0; k_base < K; k_base += kTileK) {
        const bool have_next = k_base + kTileK < K;
        RegisterStage next;
        if (have_next)
            next = prefetch(activation, weight, scales, row_base, col_base,
                            k_base + kTileK, M, N, K);
#pragma unroll
        for (int k = 0; k < kTileK; k += 16) {
            unsigned a[4];
            load_a(a, shared.operands.a + (warp_m + lane % 16) * kStride
                       + k + (lane / 16) * 8);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned b[2];
                load_b(b, shared.operands.b + (warp_n + j * 8 + lane % 8) * kStride
                           + k + ((lane / 8) % 2) * 8);
                mma(accum[j], a, b);
            }
        }
        // Every reader must finish before the single operand stage is reused.
        __syncthreads();
        if (have_next) {
            publish(shared.operands, next);
            __syncthreads();
        }
    }

    // m16n8k16 gives each lane two adjacent columns in each of two rows.
    // Direct scalar BF16 stores also cover odd N and incomplete M/N tiles.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t row = row_base + warp_m + lane / 4 + (i / 2) * 8;
            const int64_t col = col_base + warp_n + j * 8 + (lane % 4) * 2 + i % 2;
            if (row < M && col < N)
                output[row * N + col] = __float2bfloat16_rn(accum[j][i]);
        }
    }
}


// The workspace contract guarantees 4*M*K bytes, not enough for a full N*K
// BF16 weight matrix in general. Activations take the first 2*M*K bytes. The
// remaining half holds at most M decoded rows and is reused only after a GEMM
// on the same stream has consumed every row. Thus each weight is decoded once
// per invocation without hidden allocation or persistent pointer-based state.
__global__ void decode_weight_chunk(const uint8_t* __restrict__ weights,
                                     const uint8_t* __restrict__ scales,
                                     bf16* __restrict__ decoded,
                                     int64_t first_col, int64_t cols, int64_t K) {
    const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
    if (i >= cols * K) return;
    const int64_t row = first_col + i / K;
    const int64_t k = i % K;
    // K is a multiple of 32, so vectors never cross a row or scale boundary.
    const uint4 packed = *reinterpret_cast<const uint4*>(weights + row * K + k);
    const float scale = ldexpf(1.0f, int(scales[(row / 32) * (K / 32) + k / 32]) - 127);
    store_four(decoded + i,      packed.x, scale);
    store_four(decoded + i + 4,  packed.y, scale);
    store_four(decoded + i + 8,  packed.z, scale);
    store_four(decoded + i + 12, packed.w, scale);
}

constexpr int kDecodedM = 64;
constexpr int kDecodedN = 128;
constexpr int kDecodedK = 64;
constexpr int kDecodedThreads = 256;

struct __align__(32) DecodedStage {
    bf16 a[kDecodedM * kDecodedK];
    bf16 b[kDecodedN * kDecodedK];
};
struct __align__(32) DecodedStorage {
    DecodedStage stages[2];
};
static_assert(sizeof(DecodedStorage) == 48 * 1024, "Unexpected decoded shared memory");
static_assert(sizeof(DecodedStorage) <= 99 * 1024, "K2 shared-memory limit exceeded");

// XOR the eight-element vectors in a row. This avoids ldmatrix bank conflicts
// without padding, keeping both operand stages within the default 48 KB limit.
__device__ __forceinline__ int swizzled(int row, int k) {
    return row * kDecodedK + (k ^ ((row & 7) * 8));
}

__device__ __forceinline__ void copy_async(bf16* dst, const bf16* src, bool valid) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void prefetch_decoded(
    DecodedStage& stage, const bf16* activation, const bf16* weight,
    int64_t row_base, int64_t col_base, int64_t k_base,
    int64_t M, int64_t cols, int64_t K) {
#pragma unroll
    for (int i = threadIdx.x; i < kDecodedM * kDecodedK / 8; i += kDecodedThreads) {
        const int row = i / (kDecodedK / 8);
        const int k = (i % (kDecodedK / 8)) * 8;
        const bool valid = row_base + row < M && k_base + k < K;
        const bf16* src = valid ? activation + (row_base + row) * K + k_base + k : activation;
        copy_async(stage.a + swizzled(row, k), src, valid);
    }
#pragma unroll
    for (int i = threadIdx.x; i < kDecodedN * kDecodedK / 8; i += kDecodedThreads) {
        const int col = i / (kDecodedK / 8);
        const int k = (i % (kDecodedK / 8)) * 8;
        const bool valid = col_base + col < cols && k_base + k < K;
        const bf16* src = valid ? weight + (col_base + col) * K + k_base + k : weight;
        copy_async(stage.b + swizzled(col, k), src, valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void wait_decoded() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__global__ __launch_bounds__(kDecodedThreads, 2) void gemm_decoded(
    const bf16* __restrict__ activation, const bf16* __restrict__ weight,
    bf16* __restrict__ output, int64_t M, int64_t N, int64_t K, int64_t cols) {
    __shared__ DecodedStorage shared;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = (warp / 2) * 16;
    const int warp_n = (warp % 2) * 64;
    const int64_t row_base = static_cast<int64_t>(blockIdx.y) * kDecodedM;
    const int64_t col_base = static_cast<int64_t>(blockIdx.x) * kDecodedN;
    float accum[8][4] = {};

    prefetch_decoded(shared.stages[0], activation, weight, row_base, col_base, 0, M, cols, K);
    wait_decoded();
    __syncthreads();
    int stage_index = 0;
    for (int64_t k_base = 0; k_base < K; k_base += kDecodedK) {
        const bool have_next = k_base + kDecodedK < K;
        if (have_next)
            prefetch_decoded(shared.stages[stage_index ^ 1], activation, weight,
                             row_base, col_base, k_base + kDecodedK, M, cols, K);
        const DecodedStage& stage = shared.stages[stage_index];
#pragma unroll
        for (int k = 0; k < kDecodedK; k += 16) {
            unsigned a[4];
            load_a(a, stage.a + swizzled(warp_m + lane % 16, k + (lane / 16) * 8));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned b[2];
                load_b(b, stage.b + swizzled(warp_n + j * 8 + lane % 8,
                                            k + ((lane / 8) % 2) * 8));
                mma(accum[j], a, b);
            }
        }
        if (have_next) {
            wait_decoded();
            // Both the asynchronous writers and the current tile's readers
            // finish before changing stages or reusing the previous one.
            __syncthreads();
            stage_index ^= 1;
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t row = row_base + warp_m + lane / 4 + (i / 2) * 8;
            const int64_t col = col_base + warp_n + j * 8 + (lane % 4) * 2 + i % 2;
            if (row < M && col < cols)
                output[row * N + col] = __float2bfloat16_rn(accum[j][i]);
        }
    }
}

}  // namespace

void fp8_block_gemm(const bf16* x, int64_t M, int64_t K, const uint8_t* w, const uint8_t* w_scale,
                    int64_t N, bf16* y, void* workspace, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    // Activations occupy the first M*K*2 bytes of the guaranteed M*K*4 workspace.
    auto* activation = static_cast<bf16*>(workspace);
    const int64_t elements = M * K;
    if (elements > 0)
        quantize_activations<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
            x, activation, elements);
    if (M >= 256 && K > 0) {
        // Round down, never up: padding a chunk beyond M would overrun the
        // contract. The final N chunk can be short; copy_async zero fills it.
        const int64_t capacity = (M / kDecodedN) * kDecodedN;
        bf16* decoded = activation + elements;
        for (int64_t first_col = 0; first_col < N; first_col += capacity) {
            const int64_t cols = N - first_col < capacity ? N - first_col : capacity;
            const int64_t vectors = cols * K / 16;
            decode_weight_chunk<<<static_cast<unsigned>((vectors + 255) / 256), 256, 0, stream>>>(
                w, w_scale, decoded, first_col, cols, K);
            const dim3 grid(static_cast<unsigned>((cols + kDecodedN - 1) / kDecodedN),
                            static_cast<unsigned>((M + kDecodedM - 1) / kDecodedM));
            gemm_decoded<<<grid, kDecodedThreads, 0, stream>>>(activation, decoded, y + first_col, M, N, K, cols);
        }
    } else {
        const dim3 grid(static_cast<unsigned>((N + kTileN - 1) / kTileN),
                        static_cast<unsigned>((M + kTileM - 1) / kTileM));
        gemm_register_stage<<<grid, kThreads, 0, stream>>>(activation, w, w_scale, y, M, N, K);
    }
}

}  // namespace strata::ds41::kernels
