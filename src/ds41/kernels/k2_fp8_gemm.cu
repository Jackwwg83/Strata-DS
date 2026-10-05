// K2: native E4M3 tensor cores on sm_89+, exact BF16 fallback on sm_86.
// Scale each K=32 partial before accumulation: activation scales vary by row
// and K block, while weight scales vary by 32-column and K block. Two K32
// blocks share each asynchronous stage; small grids use workspace-only split K.
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"
#include <cuda_fp8.h>
#include "k2/bf16_fallback.cuh"

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
constexpr int kTileK = 64;
constexpr int kStride = 80;

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

template<int TileM, int TileN>
struct __align__(32) Stage {
    uint8_t a[TileM * kStride];
    uint8_t b[TileN * kStride];
    float a_scale[2][TileM];
    float b_scale[2][TileN / 32];
};
template<int TileM, int TileN>
struct __align__(32) SharedStorage { Stage<TileM, TileN> stages[2]; };
static_assert(sizeof(SharedStorage<64, 128>) <= 99 * 1024, "K2 shared-memory limit exceeded");
static_assert(sizeof(SharedStorage<32, 64>) <= 99 * 1024, "K2 shared-memory limit exceeded");

__device__ __forceinline__ void copy_async_16(void* dst, const void* src, bool valid) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void wait_async() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

template<int TileM, int TileN>
__device__ __forceinline__ void prefetch(
    Stage<TileM, TileN>& stage, const uint8_t* a, const float* a_scale,
    const uint8_t* b, const uint8_t* b_scale,
    int64_t m_base, int64_t n_base, int64_t k_base, int64_t k_end,
    int64_t M, int64_t N, int64_t K) {
    constexpr int threads = TileM * 4;
#pragma unroll
    for (int i = threadIdx.x; i < TileM * 4; i += threads) {
        const int row = i / 4, k = (i % 4) * 16;
        const bool valid = m_base + row < M && k_base + k < k_end;
        copy_async_16(stage.a + row * kStride + k,
                      a + (valid ? (m_base + row) * K + k_base + k : 0), valid);
    }
#pragma unroll
    for (int i = threadIdx.x; i < TileN * 4; i += threads) {
        const int col = i / 4, k = (i % 4) * 16;
        const bool valid = n_base + col < N && k_base + k < k_end;
        copy_async_16(stage.b + col * kStride + k,
                      b + (valid ? (n_base + col) * K + k_base + k : 0), valid);
    }
    if (threadIdx.x < TileM * 2) {
        const int sub = threadIdx.x / TileM, row = threadIdx.x % TileM;
        const int64_t m = m_base + row;
        const int64_t k = k_base + sub * 32;
        stage.a_scale[sub][row] = m < M && k < k_end
            ? a_scale[m * (K / 32) + k / 32] : 1.0f;
    }
    if (threadIdx.x < (TileN / 32) * 2) {
        const int sub = threadIdx.x / (TileN / 32), col = threadIdx.x % (TileN / 32);
        const int64_t n = n_base + col * 32;
        const int64_t k = k_base + sub * 32;
        stage.b_scale[sub][col] = n < N && k < k_end
            ? ldexpf(1.0f, int(b_scale[(n / 32) * (K / 32) + k / 32]) - 127)
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

template<int TileM, int TileN, bool SplitK>
__global__ __launch_bounds__(TileM * 4, 3) void gemm_fp8(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const uint8_t* __restrict__ b, const uint8_t* __restrict__ b_scale,
    bf16* __restrict__ output, float* __restrict__ partial_output,
    int64_t M, int64_t N, int64_t K, int splits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    __shared__ SharedStorage<TileM, TileN> shared;
    constexpr int fragments = TileN / 16;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = (warp / 2) * 16;
    const int warp_n = (warp % 2) * (TileN / 2);
    const int64_t m_base = static_cast<int64_t>(blockIdx.y) * TileM;
    const int64_t n_base = static_cast<int64_t>(blockIdx.x) * TileN;
    // Partition whole K32 scale blocks, including uneven partition lengths.
    const int64_t k_begin = SplitK ? ((K / 32) * blockIdx.z / splits) * 32 : 0;
    const int64_t k_end = SplitK ? ((K / 32) * (blockIdx.z + 1) / splits) * 32 : K;
    float accum[fragments][4] = {};
    if (k_begin < k_end) {
        prefetch(shared.stages[0], a, a_scale, b, b_scale,
                 m_base, n_base, k_begin, k_end, M, N, K);
        wait_async();
        __syncthreads();
    }
    int current = 0;
    for (int64_t k_base = k_begin; k_base < k_end; k_base += kTileK) {
        const int next = current ^ 1;
        const bool have_next = k_base + kTileK < k_end;
        if (have_next)
            prefetch(shared.stages[next], a, a_scale, b, b_scale,
                     m_base, n_base, k_base + kTileK, k_end, M, N, K);
        const Stage<TileM, TileN>& stage = shared.stages[current];
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            if (k_base + sub * 32 < k_end) {
                unsigned frag_a[4];
                load_a(frag_a, stage.a + (warp_m + lane % 16) * kStride
                                + sub * 32 + (lane / 16) * 16);
                const float as0 = stage.a_scale[sub][warp_m + lane / 4];
                const float as1 = stage.a_scale[sub][warp_m + lane / 4 + 8];
#pragma unroll
                for (int j = 0; j < fragments; ++j) {
                    unsigned frag_b[2];
                    load_b(frag_b, stage.b + (warp_n + j * 8 + lane % 8) * kStride
                                    + sub * 32 + ((lane / 8) % 2) * 16);
                    float partial[4];
                    mma_fp8(partial, frag_a, frag_b);
                    const float bs = stage.b_scale[sub][(warp_n + j * 8) / 32];
                    accum[j][0] = scaled_add(accum[j][0], partial[0], as0, bs);
                    accum[j][1] = scaled_add(accum[j][1], partial[1], as0, bs);
                    accum[j][2] = scaled_add(accum[j][2], partial[2], as1, bs);
                    accum[j][3] = scaled_add(accum[j][3], partial[3], as1, bs);
                }
            }
        }
        wait_async();
        __syncthreads();
        current = next;
    }
#pragma unroll
    for (int j = 0; j < fragments; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t row = m_base + warp_m + lane / 4 + (i / 2) * 8;
            const int64_t col = n_base + warp_n + j * 8 + (lane % 4) * 2 + i % 2;
            if (row < M && col < N) {
                if constexpr (SplitK)
                    partial_output[(static_cast<int64_t>(blockIdx.z) * M + row) * N + col] = accum[j][i];
                else
                    output[row * N + col] = __float2bfloat16_rn(accum[j][i]);
            }
        }
    }
#endif
}

__global__ void reduce_split_k(const float* partial, bf16* output, int64_t elements, int splits) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = partial[index];
    for (int split = 1; split < splits; ++split) value += partial[split * elements + index];
    output[index] = __float2bfloat16_rn(value);
}

template<int TileM, int TileN>
void launch_fp8(const uint8_t* a, const float* a_scale, const uint8_t* b, const uint8_t* b_scale,
                 bf16* output, float* partial, int64_t M, int64_t N, int64_t K, cudaStream_t stream) {
    const unsigned grid_m = static_cast<unsigned>((M + TileM - 1) / TileM);
    const unsigned grid_n = static_cast<unsigned>((N + TileN - 1) / TileN);
    const int64_t tiles = static_cast<int64_t>(grid_m) * grid_n;
    // 4*M*K - (M*K + M*K/8) = 23*M*K/8 spare bytes; each split
    // needs 4*M*N bytes. Never allocate, cache, or exceed caller workspace.
    const int64_t max_splits = (23 * K) / (32 * N);
    int splits = 1;
    while (splits < 8 && tiles * splits < 128 && splits * 2 <= max_splits && splits * 2 <= K / 32)
        splits *= 2;
    const dim3 grid(grid_n, grid_m, splits);
    if (splits == 1)
        gemm_fp8<TileM, TileN, false><<<grid, TileM * 4, 0, stream>>>(
            a, a_scale, b, b_scale, output, nullptr, M, N, K, 1);
    else {
        gemm_fp8<TileM, TileN, true><<<grid, TileM * 4, 0, stream>>>(
            a, a_scale, b, b_scale, output, partial, M, N, K, splits);
        const int64_t elements = M * N;
        reduce_split_k<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
            partial, output, elements, splits);
    }
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
    const cudaError_t status = cudaFuncGetAttributes(&attributes, gemm_fp8<64, 128, false>);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "K2 kernel query: %s\n", cudaGetErrorString(status));
        std::abort();
    }
    if (attributes.sharedSizeBytes < sizeof(SharedStorage<64, 128>)) {
        bf16_fallback::fp8_block_gemm(x, M, K, w, w_scale, N, y, workspace, stream);
        return;
    }
    // M*K FP8 bytes plus M*K/32 FP32 scales = 9*M*K/8 bytes. K%32
    // guarantees alignment within the promised 4*M*K-byte workspace.
    auto* quantized = static_cast<uint8_t*>(workspace);
    const int64_t elements = M * K;
    auto* scales = elements > 0 ? reinterpret_cast<float*>(quantized + elements) : nullptr;
    float* partial = elements > 0 ? scales + elements / 32 : nullptr;
    if (elements > 0)
        quantize_fp8<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
            x, quantized, scales, elements);
    if (M < 128)
        launch_fp8<32, 64>(quantized, scales, w, w_scale, y, partial, M, N, K, stream);
    else
        launch_fp8<64, 128>(quantized, scales, w, w_scale, y, partial, M, N, K, stream);
}
}  // namespace strata::ds41::kernels
