// Small-N m1 path; included inside fp8_gemv.cu's anonymous namespace.
#include "pair_decode.cuh"

template<int ROWS, bool SCALE_IN_RANGE>
__device__ __forceinline__ void accumulate_small_packet(
        const float* x, const uint4 (&packed)[ROWS], uint8_t scale, float (&acc)[ROWS]) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 a = *reinterpret_cast<const float4*>(x + j * 4);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint32_t word = j == 0 ? packed[r].x : j == 1 ? packed[r].y :
                                  j == 2 ? packed[r].z : packed[r].w;
            const float2 lo = decode_small_pair<SCALE_IN_RANGE>(uint16_t(word), scale);
            const float2 hi = decode_small_pair<SCALE_IN_RANGE>(uint16_t(word >> 16), scale);
            acc[r] = fmaf(a.x, lo.x, acc[r]);
            acc[r] = fmaf(a.y, lo.y, acc[r]);
            acc[r] = fmaf(a.z, hi.x, acc[r]);
            acc[r] = fmaf(a.w, hi.y, acc[r]);
        }
    }
}

template<int ROWS, int SPLIT>
__global__ __launch_bounds__(THREADS, 12)
void gemv_small_pair(const float* __restrict__ x, const uint8_t* __restrict__ w,
                                const uint8_t* __restrict__ scales, uint16_t* __restrict__ y,
                                int k, int n) {
    static_assert(SPLIT == 2 || SPLIT == 4, "small-N split policy");
    static_assert(ROWS == 1 || ROWS == 2, "fixed row-group policy");
    constexpr int GROUPS = THREADS / (32 * SPLIT);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int split = warp % SPLIT;
    const int group = warp / SPLIT;
    __shared__ float partial[THREADS / 32][ROWS];
    // SPLIT > 1 is selected only for n <= 8192. The grid is therefore never
    // capped, and each CTA visits its row group once; no reuse barrier is needed.
    // The 16 MiB weight bound also makes every row/column offset fit in signed int.
    const int base = int(blockIdx.x) * GROUPS * ROWS;
    const int row = base + group * ROWS;
    float acc[ROWS] = {};
    for (int col = (split * 32 + lane) * 16; col < k; col += SPLIT * 512) {
        uint4 packed[ROWS];
        const uint8_t scale = row < n ? scales[(row / 32) * (k / 32) + col / 32] : 0;
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            packed[r] = make_uint4(0, 0, 0, 0);
            if (row + r < n) packed[r] = *reinterpret_cast<const uint4*>(w + (row + r) * k + col);
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
        // Hoist the rare extreme-scale fallback out of all eight pair conversions.
        if (scale <= 246) accumulate_small_packet<ROWS, true>(x + col, packed, scale, acc);
        else accumulate_small_packet<ROWS, false>(x + col, packed, scale, acc);
#else
        accumulate_small_packet<ROWS, true>(x + col, packed, scale, acc);
#endif
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        for (int d = 16; d; d >>= 1) acc[r] += __shfl_down_sync(0xffffffffu, acc[r], d);
        if (lane == 0) partial[warp][r] = acc[r];
    }
    __syncthreads();
    if (split == 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float sum = partial[warp][r];
            #pragma unroll
            for (int z = 1; z < SPLIT; ++z) sum += partial[warp + z][r];
            if (row + r < n) y[row + r] = strata::kernels::bf16_from_f32(sum);
        }
    }
}
