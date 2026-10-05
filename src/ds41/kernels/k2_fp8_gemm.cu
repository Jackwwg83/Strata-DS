// K2-13: unscaled E4M3 values on BF16 tensor cores, on every supported SM.
// Each K32 partial is scaled before accumulation; unlike a dequantized BF16
// GEMM, no per-element power-of-two multiplication occurs on the normal path.
// Three asynchronous packed stages retain the K2-07 winner's tile layout.
// All temporary global storage comes from the caller; every launch uses stream.
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"

#include <cuda_fp8.h>

namespace strata::ds41::kernels {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kTileK = 32;
constexpr int kStages = 3;

// Match ops::act_quant_to_f32_k, including exact power-of-two boundaries.
__device__ __forceinline__ float round_pow2(float value) {
    int exponent;
    const float mantissa = frexpf(value, &exponent);
    return ldexpf(1.0f, mantissa == 0.5f ? exponent - 1 : exponent);
}

__global__ void quantize_activations(const bf16* x, bf16* quantized, float* scales, int64_t elements) {
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
    // Every finite E4M3 value, including its subnormals, is exact in BF16.
    // Keep scale separate so extreme operands can use the original FP32 path.
    quantized[index] = __float2bfloat16_rn(float(q));
    if ((threadIdx.x & 31) == 0) scales[index / 32] = scale;
}

// The XOR applies to the complete linear index, including row bit zero.
// This bijection keeps each 16-byte vector contiguous while distributing the
// eight rows of an ldmatrix across all 32 banks, without padding the K=32 tile.
__device__ __forceinline__ int swizzled(int row, int k) {
    return (row * kTileK + k) ^ ((row & 7) * 8);
}

template <int TileM, int TileN> struct __align__(32) InputStage {
    bf16 a[TileM * kTileK];
    uint8_t packed[TileN * kTileK];
    unsigned scales[TileN / 32];
    float a_scales[TileM];
};

template <int TileM, int TileN> struct __align__(32) SharedStorage {
    InputStage<TileM, TileN> stages[kStages];
    bf16 b[TileN * kTileK];
};
static_assert(sizeof(SharedStorage<64, 128>) == 33632, "Unexpected shared layout");
static_assert(sizeof(SharedStorage<64, 128>) <= 99 * 1024, "K2 shared-memory limit");

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void copy_scale_word(unsigned* dst, const uint8_t* src, int bytes) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
                 :: "r"(address), "l"(src), "r"(bytes) : "memory");
}

// Scales are copied as aligned four-byte words. A source-size operand handles
// a final partial word without reading beyond the actual scale allocation.
// Invalid M/N/K vectors are zero-filled with a safe in-bounds source pointer.
template <int TileM, int TileN, int Threads>
__device__ __forceinline__ void prefetch(
    InputStage<TileM, TileN>& stage, const bf16* activation,
    const float* a_scales, const uint8_t* weight, const uint8_t* scales, int64_t row_base,
    int64_t col_base, int64_t tile_k, int64_t M, int64_t N, int64_t K) {
    const int64_t k_base = tile_k * kTileK;
#pragma unroll
    for (int i = threadIdx.x; i < TileM * kTileK / 8; i += Threads) {
        const int row = i / 4;
        const int k = (i % 4) * 8;
        const bool valid = row_base + row < M && k_base < K;
        const bf16* src = valid ? activation + (row_base + row) * K + k_base + k : activation;
        copy16(stage.a + swizzled(row, k), src, valid);
    }
#pragma unroll
    for (int i = threadIdx.x; i < TileN * kTileK / 16; i += Threads) {
        const int col = i / 2;
        const int k = (i % 2) * 16;
        const bool valid = col_base + col < N && k_base < K;
        const uint8_t* src = valid ? weight + (col_base + col) * K + k_base + k : weight;
        copy16(stage.packed + col * kTileK + k, src, valid);
    }
    if (threadIdx.x < TileM) {
        const int64_t row = row_base + threadIdx.x;
        stage.a_scales[threadIdx.x] = row < M && k_base < K
            ? a_scales[row * (K / 32) + tile_k] : 1.0f;
    }
    if (threadIdx.x < TileN / 32) {
        const int64_t col = col_base + threadIdx.x * 32;
        const int64_t index = (col / 32) * (K / 32) + tile_k;
        const int64_t aligned = index & ~int64_t(3);
        const int64_t scale_count = ((N + 31) / 32) * (K / 32);
        const bool valid = col < N && k_base < K;
        const int64_t left = scale_count - aligned;
        const int bytes = valid ? (left >= 4 ? 4 : static_cast<int>(left)) : 0;
        copy_scale_word(stage.scales + threadIdx.x, valid ? scales + aligned : scales, bytes);
    }
    // Commit even zero-fill lookahead tiles, keeping wait_group 2 correct in
    // the prologue and epilogue as well as in the steady state.
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Decode a packed group with the CUDA FP8 conversion on every supported target.
// This conversion preserves FP8 subnormals, signed zero and NaNs. Scaling is
// deferred until a complete K32 tensor partial is available.
__device__ __forceinline__ void store_four(bf16* dst, unsigned packed) {
    __nv_fp8x4_e4m3 q;
    q.__x = packed;
    const float4 value = static_cast<float4>(q);
    auto* pair = reinterpret_cast<__nv_bfloat162*>(dst);
    pair[0] = __floats2bfloat162_rn(value.x, value.y);
    pair[1] = __floats2bfloat162_rn(value.z, value.w);
}

template <int TileM, int TileN, int Threads>
__device__ __forceinline__ void decode_stage(
    const InputStage<TileM, TileN>& stage, bf16* decoded) {
#pragma unroll
    for (int i = threadIdx.x; i < TileN * kTileK / 16; i += Threads) {
        const int col = i / 2;
        const int k = (i % 2) * 16;
        const uint4 packed = *reinterpret_cast<const uint4*>(stage.packed + col * kTileK + k);
        store_four(decoded + swizzled(col, k), packed.x);
        store_four(decoded + swizzled(col, k + 4), packed.y);
        store_four(decoded + swizzled(col, k + 8), packed.z);
        store_four(decoded + swizzled(col, k + 12), packed.w);
    }
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

// Range guard for the algebraic scale move. Nonzero unscaled products are
// multiples of 2^-18, and a K32 partial has magnitude < 2^23. A combined
// exponent in [-108,104] keeps every nonzero partial normal and finite.
// Individual operand exponents in [-117,118] likewise avoid FP32 product
// under/overflow during the reference's two operand dequantizations.
// Outside these sufficient bounds, recompute the warp's outputs below using
// exactly the original conversions and lane-wise K reduction, without BF16
// rounding of scaled operands. This also handles E8M0 byte 255 and FP8 NaNs.
__device__ __forceinline__ float partial_scale(int a_exp, int b_exp, bool& retry) {
    const int exponent = a_exp + b_exp;
    const bool safe = a_exp >= -117 && a_exp <= 118 &&
                      b_exp >= -117 && b_exp <= 118 &&
                      exponent >= -108 && exponent <= 104;
    retry |= !safe;
    return safe ? __int_as_float((exponent + 127) << 23) : 0.0f;
}

__device__ __noinline__ float reference_warp(
    const bf16* activation, const float* a_scales, const uint8_t* weight,
    const uint8_t* scales, int64_t row, int64_t col, int64_t K, int lane) {
    float sum = 0.0f;
    for (int64_t k = lane; k < K; k += 32) {
        const float a = __bfloat162float(activation[row * K + k]) *
                        a_scales[row * (K / 32) + k / 32];
        __nv_fp8_e4m3 w;
        w.__x = weight[col * K + k];
        const float b = float(w) * ldexpf(1.0f,
            int(scales[(col / 32) * (K / 32) + k / 32]) - 127);
        sum = fmaf(a, b, sum);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    return sum;
}

// Eight warps own 16x64 rectangles on the large tile. Smaller grids use
// narrower warp tiles to expose more CTAs without splitting the K reduction.
template <int TileM, int TileN, int WarpN, int Threads>
__global__ __launch_bounds__(Threads, Threads == 256 ? 2 : 5) void gemm_three_stage(
    const bf16* __restrict__ activation, const float* __restrict__ a_scales,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales, bf16* __restrict__ output,
    int64_t M, int64_t N, int64_t K) {
    static_assert(Threads == (TileM / 16) * (TileN / WarpN) * 32, "Warp layout");
    static_assert(TileM % 16 == 0 && TileN % 32 == 0 && WarpN % 8 == 0, "Tile layout");
    __shared__ SharedStorage<TileM, TileN> shared;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = (warp / (TileN / WarpN)) * 16;
    const int warp_n = (warp % (TileN / WarpN)) * WarpN;
    const int64_t row_base = static_cast<int64_t>(blockIdx.y) * TileM;
    const int64_t col_base = static_cast<int64_t>(blockIdx.x) * TileN;
    const int64_t tiles_k = K / kTileK;
    float accum[WarpN / 8][4] = {};
    bool retry = false;

    if (tiles_k > 0) {
        prefetch<TileM, TileN, Threads>(shared.stages[0], activation, a_scales, weight, scales,
                                      row_base, col_base, 0, M, N, K);
        prefetch<TileM, TileN, Threads>(shared.stages[1], activation, a_scales, weight, scales,
                                      row_base, col_base, 1, M, N, K);
    }
    int read_stage = 0;
    int write_stage = 2;
    for (int64_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
        prefetch<TileM, TileN, Threads>(shared.stages[write_stage], activation, a_scales, weight, scales,
                                      row_base, col_base, tile_k + 2, M, N, K);
        asm volatile("cp.async.wait_group 2;" ::: "memory");
        // Each thread waits for its own copies, then all consumers may read.
        __syncthreads();
        const auto& stage = shared.stages[read_stage];
        decode_stage<TileM, TileN, Threads>(stage, shared.b);
        __syncthreads();
        unsigned a[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half)
            load_a(a[half], stage.a + swizzled(warp_m + lane % 16,
                                              half * 16 + (lane / 16) * 8));
        const float as0 = stage.a_scales[warp_m + lane / 4];
        const float as1 = stage.a_scales[warp_m + lane / 4 + 8];
        const int ae0 = int((__float_as_uint(as0) >> 23) & 255u) - 127;
        const int ae1 = int((__float_as_uint(as1) >> 23) & 255u) - 127;
#pragma unroll
        for (int j = 0; j < WarpN / 8; ++j) {
            // Only these two K16 MMAs share an accumulator. Reset before the
            // next K32 block, whose activation and weight scales may differ.
            float partial[4] = {};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                unsigned b[2];
                load_b(b, shared.b + swizzled(warp_n + j * 8 + lane % 8,
                                              half * 16 + ((lane / 8) % 2) * 8));
                mma(partial, a[half], b);
            }
            const int col = warp_n + j * 8;
            const int64_t scale_index = ((col_base + col) / 32) * (K / 32) + tile_k;
            const unsigned byte = (stage.scales[col / 32] >> ((scale_index & 3) * 8)) & 255u;
            const int be = int(byte) - 127;
            const float scale0 = partial_scale(ae0, be, retry);
            const float scale1 = partial_scale(ae1, be, retry);
            accum[j][0] = fmaf(partial[0], scale0, accum[j][0]);
            accum[j][1] = fmaf(partial[1], scale0, accum[j][1]);
            accum[j][2] = fmaf(partial[2], scale1, accum[j][2]);
            accum[j][3] = fmaf(partial[3], scale1, accum[j][3]);
        }
        // Protect both the single decoded B tile and the A ring slot from
        // reuse until all tensor-core readers have finished with them.
        __syncthreads();
        read_stage = read_stage == 2 ? 0 : read_stage + 1;
        write_stage = write_stage == 2 ? 0 : write_stage + 1;
    }
    // Drain zero-fill lookahead copies before the CTA releases shared memory.
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll
    for (int j = 0; j < WarpN / 8; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) retry |= !isfinite(accum[j][i]);
    }
    // No CTA barriers remain. A whole-warp decision keeps every fallback
    // shuffle converged, even when one row/column alone needs the slow path.
    if (__any_sync(0xffffffffu, retry)) {
        for (int r = 0; r < 16; ++r) {
            const int64_t row = row_base + warp_m + r;
            if (row >= M) break;
            for (int c = 0; c < WarpN; ++c) {
                const int64_t col = col_base + warp_n + c;
                if (col >= N) break;
                const float value = reference_warp(activation, a_scales, weight,
                                                    scales, row, col, K, lane);
                if (lane == 0) output[row * N + col] = __float2bfloat16_rn(value);
            }
        }
        return;
    }
#pragma unroll
    for (int j = 0; j < WarpN / 8; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t row = row_base + warp_m + lane / 4 + (i / 2) * 8;
            const int64_t col = col_base + warp_n + j * 8 + (lane % 4) * 2 + i % 2;
            if (row < M && col < N)
                output[row * N + col] = __float2bfloat16_rn(accum[j][i]);
        }
    }
}

}  // namespace

void fp8_block_gemm(const bf16* x, int64_t M, int64_t K, const uint8_t* w, const uint8_t* w_scale,
                    int64_t N, bf16* y, void* workspace, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    // Caller storage: 2*M*K bytes of unscaled BF16 + (M*K/32)*4 bytes
    // of FP32 row/K32 scales = 17*M*K/8 = 2.125*M*K <= 4*M*K bytes.
    // K%32 aligns the scale array; no padding or extra scratch is required.
    // No allocation, deallocation, host transfer, device query, synchronization
    // or process-global state occurs, including the first/eager invocation.
    auto* activation = static_cast<bf16*>(workspace);
    const int64_t elements = M * K;
    auto* a_scales = elements > 0 ? reinterpret_cast<float*>(activation + elements) : nullptr;
    if (elements > 0)
        quantize_activations<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
            x, activation, a_scales, elements);
    const int64_t tiles_m = (M + 63) / 64;
    const int64_t tiles_n = (N + 127) / 128;
    if (tiles_m * tiles_n < 128) {
        if (M <= 128) {
            const dim3 grid(static_cast<unsigned>((N + 63) / 64),
                            static_cast<unsigned>((M + 15) / 16));
            gemm_three_stage<16, 64, 16, 128><<<grid, 128, 0, stream>>>(
                activation, a_scales, w, w_scale, y, M, N, K);
        } else {
            const dim3 grid(static_cast<unsigned>((N + 63) / 64),
                            static_cast<unsigned>((M + 31) / 32));
            gemm_three_stage<32, 64, 32, 128><<<grid, 128, 0, stream>>>(
                activation, a_scales, w, w_scale, y, M, N, K);
        }
    } else {
        const dim3 grid(static_cast<unsigned>(tiles_n), static_cast<unsigned>(tiles_m));
        gemm_three_stage<64, 128, 64, 256><<<grid, 256, 0, stream>>>(
            activation, a_scales, w, w_scale, y, M, N, K);
    }
}

}  // namespace strata::ds41::kernels
