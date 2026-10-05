// K2: three-stage asynchronous packed-input pipeline with streamed FP8 decode.
// A 64x128 output tile uses 32 accumulator floats per lane and 32.1 KiB shared.
// All temporary global storage comes from the caller; every launch uses stream.
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"

#include <cuda_fp8.h>

#include "k2/packed_decode.cuh"

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
};

template <int TileM, int TileN> struct __align__(32) SharedStorage {
    InputStage<TileM, TileN> stages[kStages];
    bf16 b[TileN * kTileK];
};
static_assert(sizeof(SharedStorage<64, 128>) == 32864, "Unexpected shared layout");
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
    const uint8_t* weight, const uint8_t* scales, int64_t row_base,
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

template <int TileM, int TileN, int Threads>
__device__ __forceinline__ void decode_stage(
    const InputStage<TileM, TileN>& stage, bf16* decoded,
    int64_t col_base, int64_t tile_k, int64_t K) {
#pragma unroll
    for (int i = threadIdx.x; i < TileN * kTileK / 16; i += Threads) {
        const int col = i / 2;
        const int k = (i % 2) * 16;
        const int64_t scale_index = ((col_base + col) / 32) * (K / 32) + tile_k;
        const unsigned byte = (stage.scales[col / 32] >> ((scale_index & 3) * 8)) & 255u;
        const uint4 packed = *reinterpret_cast<const uint4*>(stage.packed + col * kTileK + k);
        k2_detail::store_four(decoded + swizzled(col, k), packed.x, byte);
        k2_detail::store_four(decoded + swizzled(col, k + 4), packed.y, byte);
        k2_detail::store_four(decoded + swizzled(col, k + 8), packed.z, byte);
        k2_detail::store_four(decoded + swizzled(col, k + 12), packed.w, byte);
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

// Eight warps own 16x64 rectangles on the large tile. Smaller grids use
// narrower warp tiles to expose more CTAs without splitting the K reduction.
template <int TileM, int TileN, int WarpN, int Threads>
__global__ __launch_bounds__(Threads, 3) void gemm_three_stage(
    const bf16* __restrict__ activation, const uint8_t* __restrict__ weight,
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

    if (tiles_k > 0) {
        prefetch<TileM, TileN, Threads>(shared.stages[0], activation, weight, scales,
                                      row_base, col_base, 0, M, N, K);
        prefetch<TileM, TileN, Threads>(shared.stages[1], activation, weight, scales,
                                      row_base, col_base, 1, M, N, K);
    }
    int read_stage = 0;
    int write_stage = 2;
    for (int64_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
        prefetch<TileM, TileN, Threads>(shared.stages[write_stage], activation, weight, scales,
                                      row_base, col_base, tile_k + 2, M, N, K);
        asm volatile("cp.async.wait_group 2;" ::: "memory");
        // Each thread waits for its own copies, then all consumers may read.
        __syncthreads();
        const auto& stage = shared.stages[read_stage];
        decode_stage<TileM, TileN, Threads>(stage, shared.b, col_base, tile_k, K);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTileK; k += 16) {
            unsigned a[4];
            load_a(a, stage.a + swizzled(warp_m + lane % 16, k + (lane / 16) * 8));
#pragma unroll
            for (int j = 0; j < WarpN / 8; ++j) {
                unsigned b[2];
                load_b(b, shared.b + swizzled(warp_n + j * 8 + lane % 8,
                                             k + ((lane / 8) % 2) * 8));
                mma(accum[j], a, b);
            }
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
    // Only 2*M*K of the guaranteed 4*M*K bytes are needed, with no cache,
    // allocation, deallocation, host transfer or synchronization on any call.
    auto* activation = static_cast<bf16*>(workspace);
    const int64_t elements = M * K;
    if (elements > 0)
        quantize_activations<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
            x, activation, elements);
    const int64_t tiles_m = (M + 63) / 64;
    const int64_t tiles_n = (N + 127) / 128;
    if (tiles_m * tiles_n < 128) {
        if (M <= 128) {
            const dim3 grid(static_cast<unsigned>((N + 63) / 64),
                            static_cast<unsigned>((M + 15) / 16));
            gemm_three_stage<16, 64, 16, 128><<<grid, 128, 0, stream>>>(
                activation, w, w_scale, y, M, N, K);
        } else {
            const dim3 grid(static_cast<unsigned>((N + 63) / 64),
                            static_cast<unsigned>((M + 31) / 32));
            gemm_three_stage<32, 64, 32, 128><<<grid, 128, 0, stream>>>(
                activation, w, w_scale, y, M, N, K);
        }
    } else {
        const dim3 grid(static_cast<unsigned>(tiles_n), static_cast<unsigned>(tiles_m));
        gemm_three_stage<64, 128, 64, 256><<<grid, 256, 0, stream>>>(
            activation, w, w_scale, y, M, N, K);
    }
}

}  // namespace strata::ds41::kernels
