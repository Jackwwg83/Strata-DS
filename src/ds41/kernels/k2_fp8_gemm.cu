// Task K2: fuse exact block32 activation quantization into the GEMM CTA.
// Four cooperating lanes quantize each 32-value block from its register tile;
// no quantization launch or global activation workspace is used for narrow N.
// Wide N quantizes once in caller workspace instead of repeating the work in
// many column CTAs. Both paths retain one FP32 accumulator through all of K.
// No allocations, host synchronization, split-K, or capture-dependent path.
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
// Every thread loads sixteen packed B bytes. TileM also gives the per-warp
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

template <int TileM, int TileN, int Groups> struct __align__(32) Operands {
    bf16 a[Groups * TileM * kStride];
    bf16 b[TileN * kStride];
};
static_assert(sizeof(Operands<64, 128, 2>) == 20 * 1024, "Unexpected shared-memory layout");
static_assert(sizeof(Operands<32, 64, 1>) == 7680, "Unexpected shared-memory layout");
static_assert(sizeof(Operands<16, 32, 1>) == 3840, "Unexpected shared-memory layout");
static_assert(sizeof(Operands<64, 128, 2>) <= 99 * 1024, "K2 shared-memory limit exceeded");

template <int Groups> struct RegisterStage {
    uint4 a[Groups];
    uint4 b;
    float scale;
};

// K is divisible by 32, so each 16-byte vector lies within an input row. Only
// TileM*4 threads load A; every thread loads sixteen packed weights. One scale
// load per warp suffices because its sixteen adjacent columns share a 32x32 block.
template <int TileM, int Groups>
__device__ __forceinline__ RegisterStage<Groups> prefetch(
    const bf16* activation, const uint8_t* weight, const uint8_t* scales,
    int64_t row_base, int64_t col_base, int64_t k_base,
    int64_t M, int64_t N, int64_t K) {
    RegisterStage<Groups> next;
    next.b = make_uint4(0, 0, 0, 0);
    const int a_row = threadIdx.x / 4;
    const int a_k = (threadIdx.x % 4) * 8;
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        next.a[group] = make_uint4(0, 0, 0, 0);
        const int64_t row = row_base + group * TileM + a_row;
        if (threadIdx.x < TileM * 4 && row < M)
            next.a[group] = *reinterpret_cast<const uint4*>(activation + row * K + k_base + a_k);
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

// Decode eight BF16 values from a coalesced 16-byte register load. Every four
// adjacent lanes own one complete row/block32. Their two shuffles give the
// same amax as the reference's full-warp reduction without shared scratch.
__device__ __forceinline__ float4 unpack_four(unsigned lo, unsigned hi) {
    return make_float4(__uint_as_float(lo << 16), __uint_as_float(lo & 0xffff0000u),
                       __uint_as_float(hi << 16), __uint_as_float(hi & 0xffff0000u));
}

__device__ __forceinline__ float amax_four(const float4& v) {
    return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
}

__device__ __forceinline__ float4 quantize_four(float4 v, float scale) {
    v.x = fminf(fmaxf(v.x / scale, -448.0f), 448.0f);
    v.y = fminf(fmaxf(v.y / scale, -448.0f), 448.0f);
    v.z = fminf(fmaxf(v.z / scale, -448.0f), 448.0f);
    v.w = fminf(fmaxf(v.w / scale, -448.0f), 448.0f);
    // CUDA's packed conversion uses round-to-nearest-even on all targets,
    // including the software conversion fallback emitted for sm_86.
    const __nv_fp8x4_e4m3 q(v);
    return static_cast<float4>(q);
}

__device__ __forceinline__ unsigned pack_pair(float a, float b) {
    const __nv_bfloat162_raw pair = __floats2bfloat162_rn(a, b);
    return static_cast<unsigned>(pair.x) | (static_cast<unsigned>(pair.y) << 16);
}

__device__ __forceinline__ uint4 quantize_registers(uint4 raw) {
    float4 lo = unpack_four(raw.x, raw.y);
    float4 hi = unpack_four(raw.z, raw.w);
    float amax = fmaxf(amax_four(lo), amax_four(hi));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 4));
    const float scale = round_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    lo = quantize_four(lo, scale);
    hi = quantize_four(hi, scale);
    return make_uint4(pack_pair(lo.x * scale, lo.y * scale),
                      pack_pair(lo.z * scale, lo.w * scale),
                      pack_pair(hi.x * scale, hi.y * scale),
                      pack_pair(hi.z * scale, hi.w * scale));
}

template <bool Fused, int TileM, int TileN, int Groups>
__device__ __forceinline__ void publish(Operands<TileM, TileN, Groups>& shared,
                                         const RegisterStage<Groups>& next) {
    const int a_row = threadIdx.x / 4;
    const int a_k = (threadIdx.x % 4) * 8;
#pragma unroll
    for (int group = 0; group < Groups; ++group)
        if (threadIdx.x < TileM * 4) {
            // TileM*4 is a multiple of 32: every participating subgroup and
            // warp executes the shuffle, even on zero-filled M-tail rows.
            uint4 values = next.a[group];
            if constexpr (Fused) values = quantize_registers(values);
            *reinterpret_cast<uint4*>(shared.a + (group * TileM + a_row) * kStride + a_k) = values;
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

// The paired 64x128 tiles use 64 accumulator floats per lane. Both outputs
// retain FP32 sums throughout K; there are no intermediate BF16 partial sums.
// A shared B tile and each loaded B MMA fragment feed both independent groups.
// This halves weight conversion per output compared with one 64-row CTA.
template <int TileM, int TileN, int Groups, bool Fused>
__global__ __launch_bounds__(TileN * 2, (Groups > 1 ? 512 : 1024) / (TileN * 2))
void gemm_quantized_stage(
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
        publish<Fused>(shared, first);
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
                load_a(a[group], shared.a + (group * TileM + warp_m + lane % 16) * kStride
                           + k + (lane / 16) * 8);
#pragma unroll
            for (int j = 0; j < Shape::n_fragments; ++j) {
                unsigned b[2];
                load_b(b, shared.b + (warp_n + j * 8 + lane % 8) * kStride
                           + k + ((lane / 8) % 2) * 8);
#pragma unroll
                for (int group = 0; group < Groups; ++group)
                    mma(accum[group][j], a[group], b);
            }
        }
        // Every reader must finish before the single operand stage is reused.
        __syncthreads();
        if (have_next) {
            publish<Fused>(shared, next);
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
    // At most 32 medium-tile (64 tiny-tile) column CTAs repeat quantization.
    // Larger N amortizes a single activation pass much better. This dispatch
    // depends only on shape, never on graph-capture state or input contents.
    const bool fused = N <= 2048;
    const bf16* activation = x;
    if (!fused) {
        // 2*M*K bytes, within the guaranteed 4*M*K caller-owned workspace.
        auto* quantized = static_cast<bf16*>(workspace);
        const int64_t elements = M * K;
        if (elements > 0)
            quantize_activations<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
                x, quantized, elements);
        activation = quantized;
    }
    if (fused) {
        if (M <= 128) {
            const unsigned blocks = static_cast<unsigned>(((N + 31) / 32) * ((M + 15) / 16));
            gemm_quantized_stage<16, 32, 1, true><<<blocks, Tile<16, 32>::threads, 0, stream>>>(
                activation, w, w_scale, y, M, N, K);
        } else {
            const unsigned blocks = static_cast<unsigned>(((N + 63) / 64) * ((M + 31) / 32));
            gemm_quantized_stage<32, 64, 1, true><<<blocks, Tile<32, 64>::threads, 0, stream>>>(
                activation, w, w_scale, y, M, N, K);
        }
    } else {
        const int64_t tiles_n = (N + kTileN - 1) / kTileN;
        const int64_t tiles_m = (M + kRowGroups * kTileM - 1) / (kRowGroups * kTileM);
        if (tiles_m * tiles_n < 192) {
            const unsigned blocks = static_cast<unsigned>(((N + 63) / 64) * ((M + 31) / 32));
            gemm_quantized_stage<32, 64, 1, false><<<blocks, Tile<32, 64>::threads, 0, stream>>>(
                activation, w, w_scale, y, M, N, K);
        } else {
            const unsigned blocks = static_cast<unsigned>(tiles_n * tiles_m);
            gemm_quantized_stage<64, 128, 2, false><<<blocks, Tile<64, 128>::threads, 0, stream>>>(
                activation, w, w_scale, y, M, N, K);
        }
    }
}

}  // namespace strata::ds41::kernels
