// K2: native E4M3 tensor cores on sm_89+, exact BF16 fallback on sm_86.
// Scale each K=32 partial before accumulation: activation scales vary by row
// and K block, while weight scales vary by 32-column and K block.
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"
#include <cuda_fp8.h>
#include "k2/bf16_fallback.cuh"

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
constexpr int kTileM = 64;
constexpr int kTileN = 128;
constexpr int kStride = 48;
constexpr int kThreads = 256;

__device__ __forceinline__ float round_pow2(float value) {
    int exponent;
    const float mantissa = frexpf(value, &exponent);
    return ldexpf(1.0f, mantissa == 0.5f ? exponent - 1 : exponent);
}

__global__ void quantize_fp8(const bf16* x, uint8_t* quantized,
                            float* scales, int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // K % 32 == 0 makes every active warp a complete quantization block.
    if (index >= elements) return;
    const float value = __bfloat162float(x[index]);
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    const float scale = round_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    const __nv_fp8_e4m3 q(fminf(fmaxf(value / scale, -448.0f), 448.0f));
    quantized[index] = q.__x;
    if ((threadIdx.x & 31) == 0) scales[index / 32] = scale;
}

struct __align__(32) Stage {
    uint8_t a[kTileM * kStride];
    uint8_t b[kTileN * kStride];
    float a_scale[kTileM];
    float b_scale[kTileN / 32];
};
struct __align__(32) SharedStorage { Stage stages[2]; };
static_assert(sizeof(SharedStorage) <= 99 * 1024, "K2 shared-memory limit exceeded");

__device__ __forceinline__ void copy_async_16(void* dst, const void* src, bool valid) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void wait_async() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void prefetch(
    Stage& stage, const uint8_t* a, const float* a_scale,
    const uint8_t* b, const uint8_t* b_scale,
    int64_t m_base, int64_t n_base, int64_t k_base,
    int64_t M, int64_t N, int64_t K) {
    const int row = threadIdx.x / 2;
    const int k = (threadIdx.x % 2) * 16;
    if (row < kTileM) {
        const bool valid = m_base + row < M;
        copy_async_16(stage.a + row * kStride + k,
                      a + (valid ? m_base + row : 0) * K + k_base + k, valid);
    }
    const bool valid = n_base + row < N;
    copy_async_16(stage.b + row * kStride + k,
                  b + (valid ? n_base + row : 0) * K + k_base + k, valid);
    if (threadIdx.x < kTileM) {
        const int64_t m = m_base + threadIdx.x;
        stage.a_scale[threadIdx.x] = m < M ? a_scale[m * (K / 32) + k_base / 32] : 1.0f;
    }
    if (threadIdx.x < kTileN / 32) {
        const int64_t n = n_base + threadIdx.x * 32;
        stage.b_scale[threadIdx.x] = n < N
            ? ldexpf(1.0f, int(b_scale[(n / 32) * (K / 32) + k_base / 32]) - 127)
            : 1.0f;
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// ldmatrix treats each adjacent pair of FP8 bytes as one b16 item. It yields
// A={r,k; r+8,k; r,k+16; r+8,k+16}, B={column,k; column,k+16}, with four
// consecutive FP8 values per register, as required by mma.m16n8k32.
__device__ __forceinline__ void load_a(unsigned (&a)[4], const uint8_t* ptr) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(address) : "memory");
}

__device__ __forceinline__ void load_b(unsigned (&b)[2], const uint8_t* ptr) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(address) : "memory");
}

__device__ __forceinline__ void mma_fp8(float (&d)[4], const unsigned (&a)[4],
                                      const unsigned (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    // Start from zero at every scale boundary. One final scale after multiple
    // K blocks would implement a different operation.
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]), "f"(0.0f));
#endif
}

// Avoid an underflowed/overflowed intermediate product of two scales when the
// actual scaled partial is representable. The common normal-scale case is FMA.
__device__ __forceinline__ float scaled_add(float acc, float partial,
                                           float a_scale, float b_scale) {
    const float scale = a_scale * b_scale;
    if (scale >= 0x1p-126f && scale <= 0x1p127f)
        return fmaf(partial, scale, acc);
    if (isfinite(a_scale) && isfinite(b_scale)) {
        int a_exp, b_exp;
        frexpf(a_scale, &a_exp);
        frexpf(b_scale, &b_exp);
        return acc + ldexpf(partial, a_exp + b_exp - 2);
    }
    return fmaf(partial, scale, acc);
}

__global__ __launch_bounds__(kThreads, 2) void gemm_fp8(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const uint8_t* __restrict__ b, const uint8_t* __restrict__ b_scale,
    bf16* __restrict__ output, int64_t M, int64_t N, int64_t K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    __shared__ SharedStorage shared;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = (warp / 2) * 16;
    const int warp_n = (warp % 2) * 64;
    const int64_t m_base = static_cast<int64_t>(blockIdx.y) * kTileM;
    const int64_t n_base = static_cast<int64_t>(blockIdx.x) * kTileN;
    float accum[8][4] = {};
    if (K > 0) {
        prefetch(shared.stages[0], a, a_scale, b, b_scale, m_base, n_base, 0, M, N, K);
        wait_async();
        __syncthreads();
    }
    int current = 0;
    for (int64_t k_base = 0; k_base < K; k_base += 32) {
        const int next = current ^ 1;
        const bool have_next = k_base + 32 < K;
        if (have_next)
            prefetch(shared.stages[next], a, a_scale, b, b_scale,
                     m_base, n_base, k_base + 32, M, N, K);
        const Stage& stage = shared.stages[current];
        unsigned frag_a[4];
        load_a(frag_a, stage.a + (warp_m + lane % 16) * kStride + (lane / 16) * 16);
        const float as0 = stage.a_scale[warp_m + lane / 4];
        const float as1 = stage.a_scale[warp_m + lane / 4 + 8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned frag_b[2];
            load_b(frag_b, stage.b + (warp_n + j * 8 + lane % 8) * kStride
                            + ((lane / 8) % 2) * 16);
            float partial[4];
            mma_fp8(partial, frag_a, frag_b);
            const float bs = stage.b_scale[(warp_n + j * 8) / 32];
            accum[j][0] = scaled_add(accum[j][0], partial[0], as0, bs);
            accum[j][1] = scaled_add(accum[j][1], partial[1], as0, bs);
            accum[j][2] = scaled_add(accum[j][2], partial[2], as1, bs);
            accum[j][3] = scaled_add(accum[j][3], partial[3], as1, bs);
        }
        wait_async();
        __syncthreads();
        current = next;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t row = m_base + warp_m + lane / 4 + (i / 2) * 8;
            const int64_t col = n_base + warp_n + j * 8 + (lane % 4) * 2 + i % 2;
            if (row < M && col < N)
                output[row * N + col] = __float2bfloat16_rn(accum[j][i]);
        }
    }
#endif
}
}  // namespace

void fp8_block_gemm(const bf16* x, int64_t M, int64_t K, const uint8_t* w,
                    const uint8_t* w_scale, int64_t N, bf16* y,
                    void* workspace, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    // Select the loaded kernel image, not just the physical device. A binary
    // built only for sm_86 may run on a newer GPU through PTX JIT; its native
    // body is compiled out and must still use the BF16 fallback.
    cudaFuncAttributes attributes{};
    const cudaError_t status = cudaFuncGetAttributes(&attributes, gemm_fp8);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "K2 kernel query: %s\n", cudaGetErrorString(status));
        std::abort();
    }
    if (attributes.sharedSizeBytes < sizeof(SharedStorage)) {
        bf16_fallback::fp8_block_gemm(x, M, K, w, w_scale, N, y, workspace, stream);
        return;
    }
    // M*K FP8 bytes plus M*K/32 FP32 scales = 9*M*K/8 bytes. K%32
    // guarantees alignment within the promised 4*M*K-byte workspace.
    auto* quantized = static_cast<uint8_t*>(workspace);
    const int64_t elements = M * K;
    auto* scales = reinterpret_cast<float*>(quantized + elements);
    if (elements > 0)
        quantize_fp8<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
            x, quantized, scales, elements);
    const dim3 grid(static_cast<unsigned>((N + kTileN - 1) / kTileN),
                    static_cast<unsigned>((M + kTileM - 1) / kTileM));
    gemm_fp8<<<grid, kThreads, 0, stream>>>(quantized, scales, w, w_scale, y, M, N, K);
}
}  // namespace strata::ds41::kernels
