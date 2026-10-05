// Double-buffered packed weights; included inside fp8_gemv.cu's private namespace.
constexpr int ASYNC_ROWS = 4;
constexpr int ASYNC_K = 1024;
constexpr int ASYNC_VECTORS = ASYNC_K / 16;

__device__ __forceinline__ void copy_weight16(uint4* dst, const uint8_t* src, bool valid) {
#if __CUDA_ARCH__ >= 800
    const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int bytes = valid ? 16 : 0;
    // src remains a valid allocation address even for zero-fill copies. K is a
    // multiple of 32, so each in-range vector contains all 16 bytes.
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(smem), "l"(src), "r"(bytes) : "memory");
#else
    *dst = valid ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void commit_weights() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void wait_weights() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

__device__ __forceinline__ void stage_weights(
        uint4 tile[ASYNC_ROWS][ASYNC_VECTORS], const uint8_t* w,
        int64_t base, int64_t first_col, int64_t k, int64_t n) {
    // Each producer owns two disjoint vectors. Consecutive warps produce the
    // two halves of a row; a consumer warp therefore reads other warps' copies.
    #pragma unroll
    for (int v = threadIdx.x; v < ASYNC_ROWS * ASYNC_VECTORS; v += THREADS) {
        const int r = v / ASYNC_VECTORS;
        const int c = v % ASYNC_VECTORS;
        const int64_t row = base + r;
        const int64_t col = first_col + c * 16;
        const bool valid = row < n && col < k;
        const uint8_t* src = valid ? w + row * k + col : w;
        copy_weight16(&tile[r][c], src, valid);
    }
    commit_weights();
}

template<int M>
__global__ void gemv_async_weights(const float* __restrict__ x,
                                   const uint8_t* __restrict__ w,
                                   const uint8_t* __restrict__ scales,
                                   uint16_t* __restrict__ y, int64_t k, int64_t n) {
    static_assert(THREADS == ASYNC_ROWS * 32, "one consumer warp per row");
    __shared__ uint4 packed[2][ASYNC_ROWS][ASYNC_VECTORS];  // 8192 bytes
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    for (int64_t base = int64_t(blockIdx.x) * ASYNC_ROWS;
         base < n; base += int64_t(gridDim.x) * ASYNC_ROWS) {
        const int64_t row = base + warp;
        float acc[M] = {};
        stage_weights(packed[0], w, base, 0, k, n);
        wait_weights();
        // wait_group completes only this thread's copies; the CTA barrier
        // publishes every producer's completed tile to every consumer warp.
        __syncthreads();
        int buffer = 0;
        for (int64_t tile_col = 0; tile_col < k; tile_col += ASYNC_K) {
            if (tile_col + ASYNC_K < k)
                stage_weights(packed[buffer ^ 1], w, base, tile_col + ASYNC_K, k, n);

            // Decode once, then reuse the four FP32 weights for every token.
            // The lane's K order remains 16 contiguous values per 512 columns.
            #pragma unroll
            for (int v = 0; v < ASYNC_VECTORS / 32; ++v) {
                const int c = lane + v * 32;
                const int64_t col = tile_col + c * 16;
                if (row < n && col < k) {
                    const uint4 p = packed[buffer][warp][c];
                    const float sw = detail::decode_e8m0(scales[(row / 32) * (k / 32) + col / 32]);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t q = j == 0 ? p.x : j == 1 ? p.y : j == 2 ? p.z : p.w;
                        const float4 weight = make_float4(
                            detail::decode_e4m3(uint8_t(q)) * sw,
                            detail::decode_e4m3(uint8_t(q >> 8)) * sw,
                            detail::decode_e4m3(uint8_t(q >> 16)) * sw,
                            detail::decode_e4m3(uint8_t(q >> 24)) * sw);
                        #pragma unroll
                        for (int t = 0; t < M; ++t) {
                            const float4 a = *reinterpret_cast<const float4*>(
                                x + int64_t(t) * k + col + j * 4);
                            acc[t] = fmaf(a.x, weight.x, acc[t]);
                            acc[t] = fmaf(a.y, weight.y, acc[t]);
                            acc[t] = fmaf(a.z, weight.z, acc[t]);
                            acc[t] = fmaf(a.w, weight.w, acc[t]);
                        }
                    }
                }
            }
            wait_weights();
            // All consumers finish reading the old buffer before any producer
            // reuses it two tiles later. This also publishes next-tile copies.
            // Keep the final barrier: the next grid-stride row tile reuses 0.
            __syncthreads();
            buffer ^= 1;
        }
        #pragma unroll
        for (int t = 0; t < M; ++t) {
            for (int d = 16; d; d >>= 1)
                acc[t] += __shfl_down_sync(0xffffffffu, acc[t], d);
            if (lane == 0 && row < n)
                y[int64_t(t) * n + row] = strata::kernels::bf16_from_f32(acc[t]);
        }
    }
}
