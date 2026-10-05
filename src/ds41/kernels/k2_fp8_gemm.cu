// Task K2: workspace-bounded, shape-adaptive split-K for underfilled grids.
// Split CTAs use 32x128 BF16 MMA tiles and fuse exact activation quantization,
// leaving all 4*M*K caller bytes available for FP32 partials. Saturated grids
// quantize once and retain a direct-output GEMM, with no reduction launch.
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"

#include <cuda_fp8.h>

namespace strata::ds41::kernels {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kTileM = 64;
constexpr int kTileN = 256;
constexpr int kTileK = 32;
constexpr int kStride = kTileK + 8;
// Every thread loads sixteen packed B bytes. TileM also gives the per-warp
// output width: each warp owns 16xTileM values, with TileN/TileM warp columns.
template <int TileM, int TileN> struct Tile {
    static constexpr int threads = TileN * 2;
    static constexpr int warp_columns = TileN / TileM;
    static constexpr int n_fragments = TileM / 8;
    static_assert(TileM % 16 == 0 && TileN % TileM == 0, "Invalid MMA tile");
    static_assert(threads >= TileM * 4, "A tile needs more vector loaders");
};

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

template <int TileM, int TileN> struct __align__(32) Operands {
    bf16 a[TileM * kStride];
    bf16 b[TileN * kStride];
};
static_assert(sizeof(Operands<64, 256>) == 25 * 1024, "Unexpected shared-memory layout");
static_assert(sizeof(Operands<32, 128>) == 12800, "Unexpected shared-memory layout");
static_assert(sizeof(Operands<64, 256>) <= 99 * 1024, "K2 shared-memory limit exceeded");

struct RegisterStage {
    uint4 a;
    uint4 b;
    float scale;
};

// Each of eight adjacent lanes owns four contiguous activation values. The
// eight-lane maximum therefore covers exactly one 32-value quantization group.
// No BF16 rounding is introduced before FP8 rounding: the source already is
// BF16, and the reference's FP8 value times its power-of-two scale is exact.
__device__ __forceinline__ uint2 quantize_four(uint2 bits) {
    float values[4];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned word = i < 2 ? bits.x : bits.y;
        values[i] = __bfloat162float(__ushort_as_bfloat16(
            static_cast<unsigned short>(word >> ((i % 2) * 16))));
        amax = fmaxf(amax, fabsf(values[i]));
    }
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 8));
    const float scale = round_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    unsigned result[2] = {};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_fp8_e4m3 q(fminf(fmaxf(values[i] / scale, -448.0f), 448.0f));
        result[i / 2] |= static_cast<unsigned>(__bfloat16_as_ushort(
            __float2bfloat16_rn(float(q) * scale))) << ((i % 2) * 16);
    }
    return make_uint2(result[0], result[1]);
}

// The split path reads original X and quantizes directly into its register
// stage. The unsplit path vector-loads the once-quantized caller buffer.
// All B vector loads stay within a K%32==0 row; a warp's sixteen columns
// belong to one 32x32 scale block, including incomplete N tiles.
template <int TileM, bool Split>
__device__ __forceinline__ RegisterStage prefetch(
    const bf16* activation, const uint8_t* weight, const uint8_t* scales,
    int64_t row_base, int64_t col_base, int64_t k_base,
    int64_t M, int64_t N, int64_t K) {
    RegisterStage next;
    next.a = make_uint4(0, 0, 0, 0);
    next.b = make_uint4(0, 0, 0, 0);
    if constexpr (Split) {
        static_assert(TileM == 32, "Split quantization uses a 32-row tile");
        const int a_row = threadIdx.x / 8;
        const int a_k = (threadIdx.x % 8) * 4;
        const int64_t row = row_base + a_row;
        uint2 raw = make_uint2(0, 0);
        if (row < M)
            raw = *reinterpret_cast<const uint2*>(activation + row * K + k_base + a_k);
        const uint2 quantized = quantize_four(raw);
        next.a.x = quantized.x;
        next.a.y = quantized.y;
    } else {
        const int a_row = threadIdx.x / 4;
        const int a_k = (threadIdx.x % 4) * 8;
        const int64_t row = row_base + a_row;
        if (threadIdx.x < TileM * 4 && row < M)
            next.a = *reinterpret_cast<const uint4*>(activation + row * K + k_base + a_k);
    }
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

template <int TileM, int TileN, bool Split>
__device__ __forceinline__ void publish(Operands<TileM, TileN>& shared, const RegisterStage& next) {
    if constexpr (Split) {
        const int a_row = threadIdx.x / 8;
        const int a_k = (threadIdx.x % 8) * 4;
        *reinterpret_cast<uint2*>(shared.a + a_row * kStride + a_k) =
            make_uint2(next.a.x, next.a.y);
    } else {
        const int a_row = threadIdx.x / 4;
        const int a_k = (threadIdx.x % 4) * 8;
        if (threadIdx.x < TileM * 4)
            *reinterpret_cast<uint4*>(shared.a + a_row * kStride + a_k) = next.a;
    }
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

// The 32x128 split tile has eight warps and sixteen FP32 accumulators per
// lane. Every split starts and ends on a complete quantization block; no
// atomics or inter-CTA coordination is used, and each partial has one writer.
template <int TileM, int TileN, bool Split>
__global__ __launch_bounds__(TileN * 2, 1024 / (TileN * 2)) void gemm_register_stage(
    const bf16* __restrict__ activation, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales, bf16* __restrict__ output,
    float* __restrict__ partials, int64_t M, int64_t N, int64_t K, int splits) {
    using Shape = Tile<TileM, TileN>;
    static_assert(!Split || (TileM == 32 && TileN == 128), "Split tile layout");
    __shared__ Operands<TileM, TileN> shared;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = (warp / Shape::warp_columns) * 16;
    const int warp_n = (warp % Shape::warp_columns) * TileM;
    const int64_t row_base = static_cast<int64_t>(blockIdx.y) * TileM;
    const int64_t col_base = static_cast<int64_t>(blockIdx.x) * TileN;
    const int64_t split = Split ? blockIdx.z : 0;
    const int64_t k_begin = Split ? ((K / kTileK) * split / splits) * kTileK : 0;
    const int64_t k_end = Split ? ((K / kTileK) * (split + 1) / splits) * kTileK : K;
    float accum[Shape::n_fragments][4] = {};

    if (k_begin < k_end) {
        const RegisterStage first = prefetch<TileM, Split>(activation, weight, scales,
                                            row_base, col_base, k_begin, M, N, K);
        publish<TileM, TileN, Split>(shared, first);
        __syncthreads();
    }
    for (int64_t k_base = k_begin; k_base < k_end; k_base += kTileK) {
        const bool have_next = k_base + kTileK < k_end;
        RegisterStage next;
        if (have_next)
            next = prefetch<TileM, Split>(activation, weight, scales, row_base, col_base,
                            k_base + kTileK, M, N, K);
#pragma unroll
        for (int k = 0; k < kTileK; k += 16) {
            unsigned a[4];
            load_a(a, shared.a + (warp_m + lane % 16) * kStride
                       + k + (lane / 16) * 8);
#pragma unroll
            for (int j = 0; j < Shape::n_fragments; ++j) {
                unsigned b[2];
                load_b(b, shared.b + (warp_n + j * 8 + lane % 8) * kStride
                           + k + ((lane / 8) % 2) * 8);
                mma(accum[j], a, b);
            }
        }
        // Every reader must finish before the single operand stage is reused.
        __syncthreads();
        if (have_next) {
            publish<TileM, TileN, Split>(shared, next);
            __syncthreads();
        }
    }

    // m16n8k16 gives each lane two adjacent columns in each of two rows.
    // Direct scalar BF16 stores also cover odd N and incomplete M/N tiles.
#pragma unroll
    for (int j = 0; j < Shape::n_fragments; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t row = row_base + warp_m + lane / 4 + (i / 2) * 8;
            const int64_t col = col_base + warp_n + j * 8 + (lane % 4) * 2 + i % 2;
            if (row < M && col < N) {
                if constexpr (Split)
                    partials[(split * M + row) * N + col] = accum[j][i];
                else
                    output[row * N + col] = __float2bfloat16_rn(accum[j][i]);
            }
        }
    }
}

// Every output is rounded to BF16 only once, after the FP32 partial reduction.
// Partial buffers are fully overwritten before this ordered stream launch.
__global__ void reduce_partials(const float* __restrict__ partials,
                                bf16* __restrict__ output, int64_t elements, int splits) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float sum = partials[index];
    for (int split = 1; split < splits; ++split)
        sum += partials[static_cast<int64_t>(split) * elements + index];
    output[index] = __float2bfloat16_rn(sum);
}

// Select enough splits to expose roughly 192 CTAs without dividing a
// quantization group or producing tiny (<256-element) K slices. With fused
// quantization, S*M*N*sizeof(float) <= 4*M*K iff S <= K/N. The quotient form
// avoids a potentially overflowing product while computing the capacity.
int split_count(int64_t M, int64_t N, int64_t K) {
    const int64_t tiles = ((M + 31) / 32) * ((N + 127) / 128);
    int64_t splits = (192 + tiles - 1) / tiles;
    const int64_t capacity = K / N;
    const int64_t reduction_limit = K / 256;
    if (splits > capacity) splits = capacity;
    if (splits > reduction_limit) splits = reduction_limit;
    if (splits > 16) splits = 16;
    return static_cast<int>(splits >= 2 ? splits : 1);
}

}  // namespace

void fp8_block_gemm(const bf16* x, int64_t M, int64_t K, const uint8_t* w, const uint8_t* w_scale,
                    int64_t N, bf16* y, void* workspace, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    const int64_t large_m = (M + kTileM - 1) / kTileM;
    const int64_t large_n = (N + kTileN - 1) / kTileN;
    const bool full_grid = large_m * large_n >= 128;
    const int splits = full_grid ? 1 : split_count(M, N, K);
    if (splits > 1) {
        // This path uses no global activation scratch: all workspace bytes
        // belong to the bounded FP32 partial buffer. Eager/captured calls
        // execute the same two launches, with no persistent host/device state.
        auto* partials = static_cast<float*>(workspace);
        const dim3 grid(static_cast<unsigned>((N + 127) / 128),
                        static_cast<unsigned>((M + 31) / 32), splits);
        gemm_register_stage<32, 128, true><<<grid, 256, 0, stream>>>(
            x, w, w_scale, y, partials, M, N, K, splits);
        const int64_t elements = M * N;
        reduce_partials<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
            partials, y, elements, splits);
    } else {
        // Direct output avoids partial traffic and a reduction launch when
        // there are already enough output tiles. It uses only 2*M*K bytes.
        auto* activation = static_cast<bf16*>(workspace);
        const int64_t elements = M * K;
        if (elements > 0)
            quantize_activations<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
                x, activation, elements);
        if (full_grid) {
            const dim3 grid(static_cast<unsigned>(large_n), static_cast<unsigned>(large_m));
            gemm_register_stage<64, 256, false><<<grid, 512, 0, stream>>>(
                activation, w, w_scale, y, nullptr, M, N, K, 1);
        } else {
            const dim3 grid(static_cast<unsigned>((N + 127) / 128),
                            static_cast<unsigned>((M + 31) / 32));
            gemm_register_stage<32, 128, false><<<grid, 256, 0, stream>>>(
                activation, w, w_scale, y, nullptr, M, N, K, 1);
        }
    }
}

}  // namespace strata::ds41::kernels
