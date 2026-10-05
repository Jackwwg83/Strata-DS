// K2: warp-specialized producer/dequantizer and two-stage BF16 tensor-core GEMM.
// Named ready/free barriers transfer each shared-memory slot between disjoint
// producer and consumer warps. No allocation or host-side work is required.
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"

#include <cuda_fp8.h>

namespace strata::ds41::kernels {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kTileK = 32;
constexpr int kStride = 40;

__device__ __forceinline__ float round_pow2(float value) {
    int exponent;
    const float mantissa = frexpf(value, &exponent);
    return ldexpf(1.0f, mantissa == 0.5f ? exponent - 1 : exponent);
}

// A warp quantizes exactly one row's block of 32, as in ops::fp8_linear.
__global__ void quantize_activations(const bf16* x, bf16* quantized, int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;  // K % 32 makes this warp-uniform.
    const float value = __bfloat162float(x[index]);
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    const float scale = round_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    const __nv_fp8_e4m3 q(fminf(fmaxf(value / scale, -448.0f), 448.0f));
    quantized[index] = __float2bfloat16_rn(float(q) * scale);
}

template <int TileM, int TileN, int WarpN, int ProducerWarps> struct Shape {
    static constexpr int producer_threads = ProducerWarps * 32;
    static constexpr int warp_columns = TileN / WarpN;
    static constexpr int consumer_warps = TileM / 16 * warp_columns;
    static constexpr int threads = producer_threads + consumer_warps * 32;
    static constexpr int n_fragments = WarpN / 8;
    static_assert(TileM % 16 == 0 && TileN % WarpN == 0 && WarpN % 8 == 0,
                  "MMA tiles must partition the output exactly");
    static_assert(threads <= 1024 && threads % 32 == 0, "Invalid barrier count");
};

template <int TileM, int TileN> struct __align__(32) Stage {
    bf16 a[TileM * kStride];
    bf16 b[TileN * kStride];
};
static_assert(2 * sizeof(Stage<64, 128>) == 30720, "Unexpected shared-memory size");
static_assert(2 * sizeof(Stage<64, 128>) <= 48 * 1024,
              "Avoid opt-in shared memory and stay below the 99-KB task limit");

// Unaligned barrier.cta operations count all CTA threads, waiting only on the receiving
// side. Unlike legacy bar.*, these do not assert CTA-wide branch convergence.
// Each full warp takes one role; no warp executes a divergent barrier.
// IDs 1,2 are filled slots and 3,4 are consumed slots. Barrier 0 is unused.
template <int Threads>
__device__ __forceinline__ void arrive(int id) {
    asm volatile("barrier.cta.arrive %0, %1;" : : "r"(id), "n"(Threads) : "memory");
}
template <int Threads>
__device__ __forceinline__ void wait(int id) {
    asm volatile("barrier.cta.sync %0, %1;" : : "r"(id), "n"(Threads) : "memory");
}

__device__ __forceinline__ void dequantize_four(bf16* dst, unsigned packed, float scale) {
    __nv_fp8x4_e4m3 q;
    q.__x = packed;
    const float4 value = static_cast<float4>(q);
    auto* pair = reinterpret_cast<__nv_bfloat162*>(dst);
    pair[0] = __floats2bfloat162_rn(value.x * scale, value.y * scale);
    pair[1] = __floats2bfloat162_rn(value.z * scale, value.w * scale);
}

// Only producers call this function. Vectors never cross a 32-wide scale
// block or an input row. Padded rows/columns are explicitly zero-filled.
template <int TileM, int TileN, int Producers>
__device__ __forceinline__ void produce(
    Stage<TileM, TileN>& stage, const bf16* activation,
    const uint8_t* weight, const uint8_t* scales,
    int64_t row_base, int64_t col_base, int64_t k_base,
    int64_t M, int64_t N, int64_t K) {
#pragma unroll
    for (int vector = threadIdx.x; vector < TileM * 4; vector += Producers) {
        const int row = vector / 4;
        const int k = vector % 4 * 8;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (row_base + row < M)
            value = *reinterpret_cast<const uint4*>(activation + (row_base + row) * K + k_base + k);
        *reinterpret_cast<uint4*>(stage.a + row * kStride + k) = value;
    }
#pragma unroll
    for (int vector = threadIdx.x; vector < TileN * 2; vector += Producers) {
        const int col = vector / 2;
        const int k = vector % 2 * 16;
        const int64_t global_col = col_base + col;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (global_col < N)
            value = *reinterpret_cast<const uint4*>(weight + global_col * K + k_base + k);
        // Each producer warp covers 16 adjacent columns, never crossing a
        // 32-column scale block: only lane 0 needs to fetch its E8M0 scale.
        float scale = 1.0f;
        if ((threadIdx.x & 31) == 0 && global_col < N)
            scale = ldexpf(1.0f, int(scales[(global_col / 32) * (K / 32) + k_base / 32]) - 127);
        scale = __shfl_sync(0xffffffffu, scale, 0);
        bf16* dst = stage.b + col * kStride + k;
        dequantize_four(dst, value.x, scale);
        dequantize_four(dst + 4, value.y, scale);
        dequantize_four(dst + 8, value.z, scale);
        dequantize_four(dst + 12, value.w, scale);
    }
}

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
__device__ __forceinline__ void mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int TileM, int TileN, int WarpN, int ProducerWarps>
__global__ __launch_bounds__(Shape<TileM, TileN, WarpN, ProducerWarps>::threads, 2)
void gemm_warp_specialized(const bf16* __restrict__ activation,
                           const uint8_t* __restrict__ weight,
                           const uint8_t* __restrict__ scales,
                           bf16* __restrict__ output, int64_t M, int64_t N, int64_t K) {
    using S = Shape<TileM, TileN, WarpN, ProducerWarps>;
    __shared__ Stage<TileM, TileN> stages[2];
    const int64_t row_base = static_cast<int64_t>(blockIdx.y) * TileM;
    const int64_t col_base = static_cast<int64_t>(blockIdx.x) * TileN;
    const int64_t steps = K / kTileK;

    if (threadIdx.x < S::producer_threads) {
        for (int64_t step = 0; step < steps; ++step) {
            const int slot = int(step & 1);
            // Both slots start free, with no preceding consumer generation.
            // For every reuse, all consumers must finish reading the slot.
            if (step >= 2) wait<S::threads>(3 + slot);
            produce<TileM, TileN, S::producer_threads>(stages[slot], activation,
                weight, scales, row_base, col_base, step * kTileK, M, N, K);
            arrive<S::threads>(1 + slot);
        }
    } else {
        const int consumer_warp = threadIdx.x / 32 - ProducerWarps;
        const int lane = threadIdx.x & 31;
        const int warp_m = consumer_warp / S::warp_columns * 16;
        const int warp_n = consumer_warp % S::warp_columns * WarpN;
        float accum[S::n_fragments][4] = {};
        for (int64_t step = 0; step < steps; ++step) {
            const int slot = int(step & 1);
            wait<S::threads>(1 + slot);
#pragma unroll
            for (int k = 0; k < kTileK; k += 16) {
                unsigned a[4];
                load_a(a, stages[slot].a + (warp_m + lane % 16) * kStride
                           + k + lane / 16 * 8);
#pragma unroll
                for (int j = 0; j < S::n_fragments; ++j) {
                    unsigned b[2];
                    load_b(b, stages[slot].b + (warp_n + j * 8 + lane % 8) * kStride
                               + k + (lane / 8 % 2) * 8);
                    mma(accum[j], a, b);
                }
            }
            // Only signal free when another generation will reuse this slot.
            // That gives every barrier phase exactly one arrival per thread,
            // including K=0, K=32, K=64 and an odd number of K blocks.
            if (step + 2 < steps) arrive<S::threads>(3 + slot);
        }
#pragma unroll
        for (int j = 0; j < S::n_fragments; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t row = row_base + warp_m + lane / 4 + i / 2 * 8;
                const int64_t col = col_base + warp_n + j * 8 + lane % 4 * 2 + i % 2;
                if (row < M && col < N)
                    output[row * N + col] = __float2bfloat16_rn(accum[j][i]);
            }
        }
    }
}

template <int TileM, int TileN, int WarpN, int ProducerWarps>
void launch(const bf16* activation, const uint8_t* weight, const uint8_t* scales,
            bf16* output, int64_t M, int64_t N, int64_t K, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((N + TileN - 1) / TileN),
                    static_cast<unsigned>((M + TileM - 1) / TileM));
    gemm_warp_specialized<TileM, TileN, WarpN, ProducerWarps>
        <<<grid, Shape<TileM, TileN, WarpN, ProducerWarps>::threads, 0, stream>>>(
            activation, weight, scales, output, M, N, K);
}

}  // namespace

void fp8_block_gemm(const bf16* x, int64_t M, int64_t K, const uint8_t* w, const uint8_t* w_scale,
                    int64_t N, bf16* y, void* workspace, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    // Exact dequantized activations use M*K*2 bytes; the caller provides M*K*4.
    // All work, including K=0 zero output, is ordered on the supplied stream.
    auto* activation = static_cast<bf16*>(workspace);
    const int64_t elements = M * K;
    if (elements > 0)
        quantize_activations<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
            x, activation, elements);
    // Subdivide small grids for occupancy without splitting the K reduction.
    const int64_t large_tiles = ((M + 63) / 64) * ((N + 127) / 128);
    if (large_tiles < 128) {
        if (M <= 128)
            launch<16, 64, 32, 2>(activation, w, w_scale, y, M, N, K, stream);
        else
            launch<32, 64, 32, 2>(activation, w, w_scale, y, M, N, K, stream);
    } else {
        launch<64, 128, 64, 4>(activation, w, w_scale, y, M, N, K, stream);
    }
}

}  // namespace strata::ds41::kernels
