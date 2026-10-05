// Task K2: quantize each activation once, then dequantize weights into BF16
// shared-memory tiles and multiply with BF16 tensor cores / FP32 accumulators.
// Both dequantized operands are exact in BF16 (see ds41/tasks/K2.md).
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"

#include <cuda_fp8.h>
#include <mma.h>

namespace strata::ds41::kernels {
namespace {

using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 32;
constexpr int kStride = kTileK + 8;  // Keep WMMA alignment, reduce bank conflicts.
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;

// Match ops::act_quant_to_f32_k, including the power-of-two boundary rule.
__device__ __forceinline__ float round_pow2(float value) {
    int exponent;
    const float mantissa = frexpf(value, &exponent);
    return ldexpf(1.0f, mantissa == 0.5f ? exponent - 1 : exponent);
}

__global__ void quantize_activations(const bf16* x, bf16* quantized, int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // K is a multiple of 32, so this condition is uniform within each warp.
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

// E4M3 normals have three fraction bits, so applying an E8M0 power-of-two
// scale is just an exponent adjustment when the result is normal BF16.
// Keep the reference conversion for subnormal/overflow/NaN cases: this fast
// path is exact over the full input domain, not a restriction on the scales.
__device__ __forceinline__ bf16 dequantize_weight(uint8_t bits, uint8_t scale) {
    const unsigned magnitude = bits & 0x7fu;
    const int exponent = int(magnitude >> 3) + int(scale) - 7;
    if (scale != 255 && magnitude >= 8 && magnitude < 127 && exponent > 0 && exponent < 255) {
        const unsigned packed = (unsigned(bits & 0x80u) << 8) |
                                (unsigned(exponent) << 7) | ((magnitude & 7u) << 4);
        return __ushort_as_bfloat16(static_cast<unsigned short>(packed));
    }
    __nv_fp8_e4m3 q;
    q.__x = bits;
    return __float2bfloat16_rn(float(q) * ldexpf(1.0f, int(scale) - 127));
}

// One CTA computes 128x128 output values. Its eight warps each own a 32x64
// rectangle, held as eight 16x16 FP32 accumulator fragments. W is [N][K],
// which is precisely a column-major KxN operand when viewed by WMMA.
template <bool SplitK>
__global__ __launch_bounds__(kThreads) void gemm_bf16_tiles(
    const bf16* __restrict__ activation, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales, bf16* __restrict__ output,
    int64_t M, int64_t N, int64_t K, float* __restrict__ partial) {
    __shared__ __align__(32) bf16 a_tile[kTileM * kStride];
    __shared__ __align__(32) bf16 b_tile[kTileN * kStride];
    // A warp-private 16x16 tile lets us store BF16, including arbitrary M/N
    // tails, without reserving a full 128x128 FP32 output tile in shared memory.
    __shared__ __align__(32) float store_tile[kWarps * 16 * 16];
    static_assert(sizeof(a_tile) + sizeof(b_tile) + sizeof(store_tile) <= 99 * 1024,
                  "K2 shared-memory budget exceeded");

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = (warp / 2) * 32;
    const int warp_n = (warp % 2) * 64;
    const int64_t row_base = static_cast<int64_t>(blockIdx.y) * kTileM;
    const int64_t col_base = static_cast<int64_t>(blockIdx.x) * kTileN;
    const int64_t scale_stride = K / 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(accum[i][j], 0.0f);
    }

    // Split only on quantization-block boundaries. Partial sums stay FP32
    // until the final reduction, so there is still just one BF16 output round.
    int64_t k_begin = 0, k_end = K;
    if constexpr (SplitK) {
        k_begin = ((K / kTileK) * blockIdx.z / gridDim.z) * kTileK;
        k_end = ((K / kTileK) * (blockIdx.z + 1) / gridDim.z) * kTileK;
    }
    for (int64_t k_base = k_begin; k_base < k_end; k_base += kTileK) {
        // Consecutive lanes load consecutive K values. A complete warp sees
        // the same weight scale, as required by the 32x32 block layout.
#pragma unroll 4
        for (int index = threadIdx.x; index < kTileM * kTileK; index += kThreads) {
            const int row = index / kTileK;
            const int k = index % kTileK;
            const int64_t global_row = row_base + row;
            a_tile[row * kStride + k] = global_row < M
                ? activation[global_row * K + k_base + k] : __float2bfloat16_rn(0.0f);
        }
#pragma unroll 4
        for (int index = threadIdx.x; index < kTileN * kTileK; index += kThreads) {
            const int col = index / kTileK;
            const int k = index % kTileK;
            const int64_t global_col = col_base + col;
            bf16 value = __float2bfloat16_rn(0.0f);
            if (global_col < N) {
                const uint8_t bits = weight[global_col * K + k_base + k];
                const uint8_t scale = scales[(global_col / 32) * scale_stride + k_base / 32];
                value = dequantize_weight(bits, scale);
            }
            b_tile[col * kStride + k] = value;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTileK; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(a[i], a_tile + (warp_m + i * 16) * kStride + k, kStride);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
                wmma::load_matrix_sync(b, b_tile + (warp_n + j * 16) * kStride + k, kStride);
#pragma unroll
                for (int i = 0; i < 2; ++i)
                    wmma::mma_sync(accum[i][j], a[i], b, accum[i][j]);
            }
        }
        // Do not overwrite A/B until every warp has finished both K fragments.
        __syncthreads();
    }

    float* warp_store = store_tile + warp * 16 * 16;
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
                if (row < M && col < N) {
                    if constexpr (SplitK)
                        partial[(static_cast<int64_t>(blockIdx.z) * M + row) * N + col] = warp_store[index];
                    else
                        output[row * N + col] = __float2bfloat16_rn(warp_store[index]);
                }
            }
            __syncwarp();
        }
    }
}

__global__ void finish_split_k(const float* partial, bf16* output, int64_t elements, int splits) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float sum = 0.0f;
    for (int split = 0; split < splits; ++split) sum += partial[split * elements + index];
    output[index] = __float2bfloat16_rn(sum);
}

}  // namespace

void fp8_block_gemm(const bf16* x, int64_t M, int64_t K, const uint8_t* w, const uint8_t* w_scale,
                    int64_t N, bf16* y, void* workspace, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    // The first M*K*2 workspace bytes hold the exact BF16 activations.
    auto* activation = static_cast<bf16*>(workspace);
    const int64_t elements = M * K;
    if (elements > 0)
        quantize_activations<<<static_cast<unsigned>((elements + kThreads - 1) / kThreads),
                               kThreads, 0, stream>>>(x, activation, elements);
    const unsigned tiles_n = static_cast<unsigned>((N + kTileN - 1) / kTileN);
    const unsigned tiles_m = static_cast<unsigned>((M + kTileM - 1) / kTileM);
    const int64_t tiles = static_cast<int64_t>(tiles_m) * tiles_n;
    int splits = 1;
    if (tiles < 128 && K >= 256) {
        // Remaining workspace is M*K*2 bytes; each FP32 partial needs M*N*4.
        // Shape-derived limits retain the same 128x128 kernel while adding
        // independent CTAs when a small output otherwise leaves most SMs idle.
        const int64_t capacity = K / (2 * N);
        const int64_t wanted = (128 + tiles - 1) / tiles;
        int64_t count = capacity < wanted ? capacity : wanted;
        if (count > 8) count = 8;
        if (count > K / kTileK) count = K / kTileK;
        if (count > 1) splits = static_cast<int>(count);
    }
    const dim3 grid(tiles_n, tiles_m, splits);
    if (splits == 1) {
        gemm_bf16_tiles<false><<<grid, kThreads, 0, stream>>>(activation, w, w_scale, y, M, N, K, nullptr);
    } else {
        float* partial = reinterpret_cast<float*>(activation + elements);
        gemm_bf16_tiles<true><<<grid, kThreads, 0, stream>>>(activation, w, w_scale, y, M, N, K, partial);
        const int64_t outputs = M * N;
        finish_split_k<<<static_cast<unsigned>((outputs + kThreads - 1) / kThreads),
                         kThreads, 0, stream>>>(partial, y, outputs, splits);
    }
}

}  // namespace strata::ds41::kernels
