// sm_86 fallback: exact BF16 operands, two cp.async stages.
#pragma once
// Task K2: 128x256 BF16 tensor-core GEMM, FP32 accumulation, two async stages.
// Activations are quantized once. Each stage copies packed FP8 weights and BF16
// activations with cp.async, then dequantizes the weights before MMA consumes it.
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"

#include <cuda_fp8.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

namespace strata::ds41::kernels::bf16_fallback {
namespace {

using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;
constexpr int kTileM = 128;
constexpr int kTileN = 256;
constexpr int kTileK = 32;
constexpr int kStride = kTileK + 8;
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;

// Preserve the reference's exact power-of-two boundary and E4M3 rounding.
__device__ __forceinline__ float round_pow2(float value) {
    int exponent;
    const float mantissa = frexpf(value, &exponent);
    return ldexpf(1.0f, mantissa == 0.5f ? exponent - 1 : exponent);
}

__global__ void quantize_activations(const bf16* x, bf16* quantized, int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // The interface guarantees K % 32 == 0; this is a whole-warp condition.
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

struct __align__(32) Stage {
    bf16 a[kTileM * kStride];
    bf16 b[kTileN * kStride];
    uint8_t packed_b[kTileN * kTileK];
    float scales[kTileN / 32];
};

// The epilogue reuses the stages after a CTA-wide barrier. 76 KiB + 64 bytes,
// including both packed-weight staging buffers, fits the 99 KiB task limit.
union __align__(32) SharedStorage {
    Stage stages[2];
    float output[kWarps * 16 * 16];
};
static_assert(sizeof(SharedStorage) == 76 * 1024 + 64, "Unexpected shared-memory layout");
static_assert(sizeof(SharedStorage) <= 99 * 1024, "K2 shared-memory limit exceeded");

__device__ __forceinline__ void copy_async_16(void* dst, const void* src, bool valid) {
    const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int bytes = valid ? 16 : 0;
    // src-size=0 zero-fills a tail without accessing its global source.
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void commit_async() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void wait_async() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// Exactly one 16-byte activation copy and one packed-weight copy per thread.
// Select an in-bounds source even for zero-fill copies, avoiding invalid pointer
// arithmetic for partial output tiles. All destinations and sources are aligned.
__device__ __forceinline__ void prefetch_stage(
    Stage& stage, const bf16* activation, const uint8_t* weight, const uint8_t* scales,
    int64_t row_base, int64_t col_base, int64_t k_base,
    int64_t M, int64_t N, int64_t K) {
    const int a_row = threadIdx.x / 4;
    const int a_k = (threadIdx.x % 4) * 8;
    const int64_t global_row = row_base + a_row;
    const bool valid_a = global_row < M;
    const bf16* a_src = activation + (valid_a ? global_row : 0) * K + k_base + a_k;
    copy_async_16(stage.a + a_row * kStride + a_k, a_src, valid_a);

    const int b_col = threadIdx.x / 2;
    const int b_k = (threadIdx.x % 2) * 16;
    const int64_t global_col = col_base + b_col;
    const bool valid_b = global_col < N;
    const uint8_t* b_src = weight + (valid_b ? global_col : 0) * K + k_base + b_k;
    copy_async_16(stage.packed_b + b_col * kTileK + b_k, b_src, valid_b);
    // One scale is shared by 32 consecutive weight columns. Cache it once per
    // stage instead of recomputing ldexpf in all 512 dequantization threads.
    if (threadIdx.x < kTileN / 32) {
        const int64_t scale_col = col_base + threadIdx.x * 32;
        stage.scales[threadIdx.x] = scale_col < N
            ? ldexpf(1.0f, int(scales[(scale_col / 32) * (K / 32) + k_base / 32]) - 127)
            : 1.0f;
    }
    commit_async();
}

__device__ __forceinline__ void dequantize_stage(Stage& stage) {
    // Consecutive lanes handle four consecutive K elements. The packed loads
    // and paired BF16 stores are coalesced, rather than one scalar store per
    // lane at widely separated column offsets. FP8 pairs convert through exact
    // FP16 representations, then scale in FP32 before the final BF16 rounding.
#pragma unroll
    for (int index = threadIdx.x * 4; index < kTileN * kTileK; index += kThreads * 4) {
        const int col = index / kTileK;
        const int k = index % kTileK;
        const unsigned packed = *reinterpret_cast<const unsigned*>(stage.packed_b + index);
        const float scale = stage.scales[col / 32];
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            __nv_fp8x2_e4m3 q;
            q.__x = static_cast<unsigned short>(packed >> (pair * 16));
            const float2 values = static_cast<float2>(q);
            const __nv_bfloat162 result = __floats2bfloat162_rn(values.x * scale, values.y * scale);
            *reinterpret_cast<__nv_bfloat162*>(stage.b + col * kStride + k + pair * 2) = result;
        }
    }
}

// Sixteen warps form a 4x4 grid of 32x64 output rectangles. Each warp holds
// eight 16x16 FP32 fragments. Packed W[N][K] becomes a column-major KxN operand.
__global__ __launch_bounds__(kThreads) void gemm_two_stage(
    const bf16* __restrict__ activation, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales, bf16* __restrict__ output,
    int64_t M, int64_t N, int64_t K) {
    extern __shared__ __align__(32) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<SharedStorage*>(shared_bytes);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = (warp / 4) * 32;
    const int warp_n = (warp % 4) * 64;
    const int64_t row_base = static_cast<int64_t>(blockIdx.y) * kTileM;
    const int64_t col_base = static_cast<int64_t>(blockIdx.x) * kTileN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(accum[i][j], 0.0f);
    }

    if (K > 0) {
        prefetch_stage(shared.stages[0], activation, weight, scales,
                       row_base, col_base, 0, M, N, K);
        wait_async();
        __syncthreads();
        dequantize_stage(shared.stages[0]);
        __syncthreads();
    }

    int current = 0;
    for (int64_t k_base = 0; k_base < K; k_base += kTileK) {
        const int next = current ^ 1;
        const bool have_next = k_base + kTileK < K;
        // The previous end-of-iteration barrier releases this alternate stage.
        // Global copies proceed while tensor cores consume the current stage.
        if (have_next)
            prefetch_stage(shared.stages[next], activation, weight, scales,
                           row_base, col_base, k_base + kTileK, M, N, K);
        const Stage& stage = shared.stages[current];
#pragma unroll
        for (int k = 0; k < kTileK; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(a[i], stage.a + (warp_m + i * 16) * kStride + k, kStride);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
                wmma::load_matrix_sync(b, stage.b + (warp_n + j * 16) * kStride + k, kStride);
#pragma unroll
                for (int i = 0; i < 2; ++i)
                    wmma::mma_sync(accum[i][j], a[i], b, accum[i][j]);
            }
        }
        // wait_group is per thread; the CTA barrier makes all copied bytes
        // visible and also ensures no warp still reads the old stage.
        wait_async();
        __syncthreads();
        if (have_next) {
            dequantize_stage(shared.stages[next]);
            __syncthreads();
        }
        current = next;
    }

    // Covers K=0 as well, before the staging union becomes warp-local output.
    __syncthreads();
    float* warp_store = shared.output + warp * 16 * 16;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::store_matrix_sync(warp_store, accum[i][j], 16, wmma::mem_row_major);
            __syncwarp();
#pragma unroll
            for (int index = lane; index < 16 * 16; index += 32) {
                const int64_t row = row_base + warp_m + i * 16 + index / 16;
                const int64_t col = col_base + warp_n + j * 16 + index % 16;
                if (row < M && col < N)
                    output[row * N + col] = __float2bfloat16_rn(warp_store[index]);
            }
            __syncwarp();
        }
    }
}

}  // namespace

void fp8_block_gemm(const bf16* x, int64_t M, int64_t K, const uint8_t* w, const uint8_t* w_scale,
                    int64_t N, bf16* y, void* workspace, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    // Uses M*K*2 of the guaranteed M*K*4 workspace bytes, on the supplied stream.
    auto* activation = static_cast<bf16*>(workspace);
    const int64_t elements = M * K;
    if (elements > 0)
        quantize_activations<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
            x, activation, elements);
    const cudaError_t attr = cudaFuncSetAttribute(gemm_two_stage,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage));
    if (attr != cudaSuccess) {
        std::fprintf(stderr, "K2 shared-memory opt-in: %s\n", cudaGetErrorString(attr));
        std::abort();
    }
    const dim3 grid(static_cast<unsigned>((N + kTileN - 1) / kTileN),
                    static_cast<unsigned>((M + kTileM - 1) / kTileM));
    gemm_two_stage<<<grid, kThreads, sizeof(SharedStorage), stream>>>(
        activation, w, w_scale, y, M, N, K);
}

}  // namespace strata::ds41::kernels::bf16_fallback
