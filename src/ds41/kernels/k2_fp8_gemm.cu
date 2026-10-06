// Task K2: grouped output tiles share one dequantized weight stage.
// The grouped path publishes eight decoded weights per aligned 16-byte store.
// Each large CTA keeps two 64x128 outputs live through the K loop: a weight
// tile is fetched/converted once and its MMA registers feed both row groups.
// Four neighboring CTA row groups are scheduled together for L2 weight reuse.
// Small grids retain conventional narrow output tiles without splitting K.
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"

#include <cuda_fp8.h>

namespace strata::ds41::kernels {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kTileM = 64;
constexpr int kTileN = 128;
constexpr int kRowGroups = 2;
constexpr int kTileK = 32;
constexpr int kStride = kTileK + 8;
// Every thread loads sixteen packed B bytes in total. TileM also gives the per-warp
// output width: each warp owns 16xTileM values per group, with TileN/TileM warp columns.
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

// Only grouped CTAs use the compact XOR mapping. Apply XOR to the entire
// linear index: it permutes a row bit too, while preserving each aligned K8.
// Narrow CTAs retain the control's padded layout and producer ownership.
template <int Groups>
__device__ __forceinline__ int shared_index(int row, int k) {
    if constexpr (Groups > 1)
        return (row * kTileK + k) ^ ((row & 7) * 8);
    else
        return row * kStride + k;
}

template <int TileM, int TileN, int Groups> struct __align__(32) Operands {
    static constexpr int stride = Groups > 1 ? kTileK : kStride;
    bf16 a[Groups * TileM * stride];
    bf16 b[TileN * stride];
};
static_assert(sizeof(Operands<64, 128, 2>) == 16 * 1024, "Unexpected shared-memory layout");
static_assert(sizeof(Operands<32, 64, 1>) == 7680, "Unexpected shared-memory layout");
static_assert(sizeof(Operands<16, 64, 1>) == 6400, "Unexpected shared-memory layout");
static_assert(sizeof(Operands<64, 128, 2>) <= 99 * 1024, "K2 shared-memory limit exceeded");

template <int Groups> struct RegisterStage {
    uint4 a[Groups];
    uint4 b;
    float scale;
};

// Grouped threads use two eight-FP8 packets rather than one sixteen-FP8
// packet. Four lanes cover K32 for each column; each round spans eight columns
// per warp, within one scale block. Packed B still uses four words per lane.
template <> struct RegisterStage<2> {
    uint4 a[2];
    uint2 b[2];
    float scale[2];
};

// K is divisible by 32, so every input vector lies within one row. A ownership
// and the one-stage register lookahead schedule are unchanged. Narrow CTAs
// keep their original sixteen-byte B vector and one scale per warp.
template <int TileM, int Groups>
__device__ __forceinline__ RegisterStage<Groups> prefetch(
    const bf16* activation, const uint8_t* weight, const uint8_t* scales,
    int64_t row_base, int64_t col_base, int64_t k_base,
    int64_t M, int64_t N, int64_t K) {
    RegisterStage<Groups> next;
    const int a_row = threadIdx.x / 4;
    const int a_k = (threadIdx.x % 4) * 8;
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        next.a[group] = make_uint4(0, 0, 0, 0);
        const int64_t row = row_base + group * TileM + a_row;
        if (threadIdx.x < TileM * 4 && row < M)
            next.a[group] = *reinterpret_cast<const uint4*>(activation + row * K + k_base + a_k);
    }
    if constexpr (Groups > 1) {
        static_assert(Groups == 2 && TileM == 64, "Unexpected grouped shape");
        const int b_k = (threadIdx.x % 4) * 8;
#pragma unroll
        for (int packet = 0; packet < 2; ++packet) {
            const int64_t col = col_base + threadIdx.x / 4 + packet * (kTileN / 2);
            next.b[packet] = make_uint2(0, 0);
            if (col < N)
                next.b[packet] = *reinterpret_cast<const uint2*>(weight + col * K + k_base + b_k);
            float scale = 1.0f;
            if ((threadIdx.x % 32) == 0 && col < N)
                scale = ldexpf(1.0f, int(scales[(col / 32) * (K / 32) + k_base / 32]) - 127);
            next.scale[packet] = __shfl_sync(0xffffffffu, scale, 0);
        }
    } else {
        next.b = make_uint4(0, 0, 0, 0);
        const int b_col = threadIdx.x / 2;
        const int b_k = (threadIdx.x % 2) * 16;
        const int64_t col = col_base + b_col;
        if (col < N)
            next.b = *reinterpret_cast<const uint4*>(weight + col * K + k_base + b_k);
        float scale = 1.0f;
        if ((threadIdx.x % 32) == 0 && col < N)
            scale = ldexpf(1.0f, int(scales[(col / 32) * (K / 32) + k_base / 32]) - 127);
        next.scale = __shfl_sync(0xffffffffu, scale, 0);
    }
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

// Keep CUDA FP8 -> FP32, FP32 multiply, then round-to-nearest BF16 pairs.
// The raw pair conversion just packages the same four BF16 pair results.
__device__ __forceinline__ unsigned scaled_pair(float x, float y, float scale) {
    const __nv_bfloat162_raw pair = __floats2bfloat162_rn(x * scale, y * scale);
    return static_cast<unsigned>(pair.x) | (static_cast<unsigned>(pair.y) << 16);
}

__device__ __forceinline__ void store_eight(bf16* dst, uint2 packed, float scale) {
    __nv_fp8x4_e4m3 q0, q1;
    q0.__x = packed.x;
    q1.__x = packed.y;
    const float4 v0 = static_cast<float4>(q0);
    const float4 v1 = static_cast<float4>(q1);
    const unsigned x = scaled_pair(v0.x, v0.y, scale);
    const unsigned y = scaled_pair(v0.z, v0.w, scale);
    const unsigned z = scaled_pair(v1.x, v1.y, scale);
    const unsigned w = scaled_pair(v1.z, v1.w, scale);
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "r"(address), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

template <int TileM, int TileN, int Groups>
__device__ __forceinline__ void publish(Operands<TileM, TileN, Groups>& shared,
                                         const RegisterStage<Groups>& next) {
    const int a_row = threadIdx.x / 4;
    const int a_k = (threadIdx.x % 4) * 8;
#pragma unroll
    for (int group = 0; group < Groups; ++group)
        if (threadIdx.x < TileM * 4)
            *reinterpret_cast<uint4*>(shared.a + shared_index<Groups>(group * TileM + a_row, a_k)) = next.a[group];
    if constexpr (Groups > 1) {
        static_assert(TileN == kTileN, "Unexpected grouped width");
        const int b_k = (threadIdx.x % 4) * 8;
#pragma unroll
        for (int packet = 0; packet < 2; ++packet) {
            const int b_col = threadIdx.x / 4 + packet * (TileN / 2);
            store_eight(shared.b + shared_index<Groups>(b_col, b_k), next.b[packet], next.scale[packet]);
        }
    } else {
        const int b_col = threadIdx.x / 2;
        const int b_k = (threadIdx.x % 2) * 16;
        bf16* dst = shared.b + b_col * kStride + b_k;
        store_four(dst,      next.b.x, next.scale);
        store_four(dst + 4,  next.b.y, next.scale);
        store_four(dst + 8,  next.b.z, next.scale);
        store_four(dst + 12, next.b.w, next.scale);
    }
}

// Operand register mappings follow PTX mma.m16n8k16: A is four packed BF16
// registers; B is two. Both layouts preserve contiguous aligned K8 fragments,
// so neither ldmatrix uses .trans. B is logically column-major KxN.
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

// The paired 64x128 tiles use 64 accumulator floats per lane. Both outputs
// retain FP32 sums throughout K; there are no intermediate BF16 partial sums.
// A shared B tile and each loaded B MMA fragment feed both independent groups.
// This halves weight conversion per output compared with one 64-row CTA.
template <int TileM, int TileN, int Groups>
__global__ __launch_bounds__(TileN * 2, (Groups > 1 ? 512 : 1024) / (TileN * 2))
void gemm_grouped_stage(
    const bf16* __restrict__ activation, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales, bf16* __restrict__ output,
    int64_t M, int64_t N, int64_t K) {
    using Shape = Tile<TileM, TileN>;
    __shared__ Operands<TileM, TileN, Groups> shared;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = (warp / Shape::warp_columns) * 16;
    const int warp_n = (warp % Shape::warp_columns) * TileM;
    const int64_t tiles_m = (M + Groups * TileM - 1) / (Groups * TileM);
    const int64_t tiles_n = (N + TileN - 1) / TileN;
    // A bijective grouped traversal, including the last short M supergroup.
    // Adjacent CTAs share an N tile before moving to the next weight slab.
    constexpr int kGroupCTAs = 4;
    const int64_t supergroup = blockIdx.x / (kGroupCTAs * tiles_n);
    const int64_t first_m = supergroup * kGroupCTAs;
    const int group_m = static_cast<int>(tiles_m - first_m < kGroupCTAs
                                        ? tiles_m - first_m : kGroupCTAs);
    const int64_t within = blockIdx.x % (kGroupCTAs * tiles_n);
    const int64_t row_base = (first_m + within % group_m) * Groups * TileM;
    const int64_t col_base = (within / group_m) * TileN;
    float accum[Groups][Shape::n_fragments][4] = {};

    if (K > 0) {
        const RegisterStage<Groups> first = prefetch<TileM, Groups>(activation, weight, scales,
                                            row_base, col_base, 0, M, N, K);
        publish(shared, first);
        __syncthreads();
    }
    for (int64_t k_base = 0; k_base < K; k_base += kTileK) {
        const bool have_next = k_base + kTileK < K;
        RegisterStage<Groups> next;
        if (have_next)
            next = prefetch<TileM, Groups>(activation, weight, scales, row_base, col_base,
                            k_base + kTileK, M, N, K);
#pragma unroll
        for (int k = 0; k < kTileK; k += 16) {
            unsigned a[Groups][4];
#pragma unroll
            for (int group = 0; group < Groups; ++group)
                load_a(a[group], shared.a + shared_index<Groups>(group * TileM + warp_m + lane % 16,
                           k + (lane / 16) * 8));
#pragma unroll
            for (int j = 0; j < Shape::n_fragments; ++j) {
                unsigned b[2];
                load_b(b, shared.b + shared_index<Groups>(warp_n + j * 8 + lane % 8,
                           k + ((lane / 8) % 2) * 8));
#pragma unroll
                for (int group = 0; group < Groups; ++group)
                    mma(accum[group][j], a[group], b);
            }
        }
        // Every reader must finish before the single operand stage is reused.
        __syncthreads();
        if (have_next) {
            publish(shared, next);
            __syncthreads();
        }
    }

    // m16n8k16 gives each lane two adjacent columns in each of two rows.
    // Guarded stores cover incomplete M groups and arbitrary N tails.
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
#pragma unroll
        for (int j = 0; j < Shape::n_fragments; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t row = row_base + group * TileM + warp_m + lane / 4 + (i / 2) * 8;
                const int64_t col = col_base + warp_n + j * 8 + (lane % 4) * 2 + i % 2;
                if (row < M && col < N)
                    output[row * N + col] = __float2bfloat16_rn(accum[group][j][i]);
            }
        }
    }
}

}  // namespace

void fp8_block_gemm(const bf16* x, int64_t M, int64_t K, const uint8_t* w, const uint8_t* w_scale,
                    int64_t N, bf16* y, void* workspace, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    // Only M*K*2 bytes of the guaranteed M*K*4 workspace are used.
    auto* activation = static_cast<bf16*>(workspace);
    const int64_t elements = M * K;
    if (elements > 0)
        quantize_activations<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
            x, activation, elements);
    const int64_t tiles_n = (N + kTileN - 1) / kTileN;
    const int64_t tiles_m = (M + kRowGroups * kTileM - 1) / (kRowGroups * kTileM);
    // Narrow output grids need more CTAs than the paired layout can supply.
    // Keep one output tile per CTA here; no split-K or additional workspace.
    if (tiles_m * tiles_n < 192) {
        if (M <= 128) {
            const unsigned blocks = static_cast<unsigned>(((N + 63) / 64) * ((M + 15) / 16));
            gemm_grouped_stage<16, 64, 1><<<blocks, Tile<16, 64>::threads, 0, stream>>>(
                activation, w, w_scale, y, M, N, K);
        } else {
            const unsigned blocks = static_cast<unsigned>(((N + 63) / 64) * ((M + 31) / 32));
            gemm_grouped_stage<32, 64, 1><<<blocks, Tile<32, 64>::threads, 0, stream>>>(
                activation, w, w_scale, y, M, N, K);
        }
    } else {
        const unsigned blocks = static_cast<unsigned>(tiles_n * tiles_m);
        gemm_grouped_stage<64, 128, 2><<<blocks, Tile<64, 128>::threads, 0, stream>>>(
            activation, w, w_scale, y, M, N, K);
    }
}

}  // namespace strata::ds41::kernels
