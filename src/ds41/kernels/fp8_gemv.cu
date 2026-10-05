// src/ds41/kernels/fp8_gemv.cu - FP8 block-scaled GEMV without FP8 tensor cores.
#include "strata/ds41/fp8_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::ds41 {
namespace {

constexpr int THREADS = 128;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fp8_block_gemv %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void validate(const void* x, int m, int64_t k) {
    if (!x || m < 1 || m > 8 || k <= 0 || k % 32 != 0 ||
        uint64_t(k) > std::numeric_limits<size_t>::max() / (8 * sizeof(float)))
        throw std::invalid_argument("fp8_block_gemv: need x, 1 <= m <= 8, positive k divisible by 32");
}

// One warp quantizes one block. Production and debug calls execute exactly this arithmetic.
__global__ void quantize(const uint16_t* __restrict__ x, int64_t blocks,
                         float* __restrict__ dequant, uint8_t* __restrict__ q,
                         float* __restrict__ scales) {
    const int lane = threadIdx.x & 31;
    for (int64_t b = int64_t(blockIdx.x) * (THREADS / 32) + threadIdx.x / 32;
         b < blocks; b += int64_t(gridDim.x) * (THREADS / 32)) {
        const int64_t i = b * 32 + lane;
        const float v = strata::kernels::f32_from_bf16(x[i]);
        float amax = fmaxf(fabsf(v), 1e-4f);
        for (int d = 16; d; d >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
        const float s = detail::activation_scale(amax);
        const uint8_t byte = detail::encode_e4m3(v / s);
        if (dequant) dequant[i] = detail::decode_e4m3(byte) * s;
        if (q) q[i] = byte;
        if (scales && lane == 0) scales[b] = s;
    }
}

// SPLIT warps cooperate on ROWS output rows. Small N gets more independent K slices;
// adjacent output rows reuse each float4 activation load and have independent accumulators.
// Every weight vector is loaded and decoded once for all M activation rows.
template<int M, int ROWS, int SPLIT, bool WIDE>
__global__ void gemv(const float* __restrict__ x, const uint8_t* __restrict__ w,
                     const uint8_t* __restrict__ scales, uint16_t* __restrict__ y,
                     int64_t k, int64_t n) {
    constexpr int GROUPS = THREADS / (32 * SPLIT);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int split = warp % SPLIT;
    const int group = warp / SPLIT;
    __shared__ float partial[THREADS / 32][ROWS][M];
    for (int64_t base = int64_t(blockIdx.x) * GROUPS * ROWS;
         base < n; base += int64_t(gridDim.x) * GROUPS * ROWS) {
        const int64_t row = base + group * ROWS;
        float acc[ROWS][M] = {};
        for (int64_t col = int64_t(split * 32 + lane) * 16; col < k; col += SPLIT * 512) {
            uint4 packed[ROWS];
            // ROWS is 1 or 2 and row is a multiple of ROWS, so the group never crosses a scale row.
            const float sw = row < n ? detail::decode_e8m0(scales[(row / 32) * (k / 32) + col / 32]) : 0;
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                packed[r] = make_uint4(0, 0, 0, 0);
                if (row + r < n) {
                    const uint8_t* p = w + (row + r) * k + col;
                    if constexpr (WIDE) {
                        packed[r] = *reinterpret_cast<const uint4*>(p);
                    } else {
                        uint32_t words[4] = {};
                        #pragma unroll
                        for (int j = 0; j < 16; ++j) words[j / 4] |= uint32_t(p[j]) << ((j % 4) * 8);
                        packed[r] = make_uint4(words[0], words[1], words[2], words[3]);
                    }
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float4 weight[ROWS];
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    const uint32_t word = j == 0 ? packed[r].x : j == 1 ? packed[r].y :
                                          j == 2 ? packed[r].z : packed[r].w;
                    weight[r] = make_float4(detail::decode_e4m3(uint8_t(word)) * sw,
                                            detail::decode_e4m3(uint8_t(word >> 8)) * sw,
                                            detail::decode_e4m3(uint8_t(word >> 16)) * sw,
                                            detail::decode_e4m3(uint8_t(word >> 24)) * sw);
                }
                #pragma unroll
                for (int t = 0; t < M; ++t) {
                    const float* p = x + int64_t(t) * k + col + j * 4;
                    float4 a;
                    if constexpr (WIDE) a = *reinterpret_cast<const float4*>(p);
                    else a = make_float4(p[0], p[1], p[2], p[3]);
                    #pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        acc[r][t] = fmaf(a.x, weight[r].x, acc[r][t]);
                        acc[r][t] = fmaf(a.y, weight[r].y, acc[r][t]);
                        acc[r][t] = fmaf(a.z, weight[r].z, acc[r][t]);
                        acc[r][t] = fmaf(a.w, weight[r].w, acc[r][t]);
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            #pragma unroll
            for (int t = 0; t < M; ++t) {
                for (int d = 16; d; d >>= 1) acc[r][t] += __shfl_down_sync(0xffffffffu, acc[r][t], d);
                if (lane == 0) {
                    if constexpr (SPLIT == 1) {
                        if (row + r < n) y[int64_t(t) * n + row + r] = strata::kernels::bf16_from_f32(acc[r][t]);
                    } else {
                        partial[warp][r][t] = acc[r][t];
                    }
                }
            }
        }
        if constexpr (SPLIT > 1) {
            __syncthreads();
            if (split == 0 && lane == 0) {
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    #pragma unroll
                    for (int t = 0; t < M; ++t) {
                        float sum = partial[warp][r][t];
                        #pragma unroll
                        for (int z = 1; z < SPLIT; ++z) sum += partial[warp + z][r][t];
                        if (row + r < n) y[int64_t(t) * n + row + r] = strata::kernels::bf16_from_f32(sum);
                    }
                }
            }
            // All warps must finish reading before a grid-stride iteration overwrites partials.
            __syncthreads();
        }
    }
}


// A bounded activation tile is shared by every output-row group in the CTA. Weights
// stay in registers and each decoded value is reused for all M tokens. M=1 and small
// N keep the baseline path: they have little CTA reuse to repay the tile barriers.
constexpr int ACT_TILE = 512;

// Float4-vector swizzle: contiguous staging stores and strided warp reads both use
// all shared-memory banks. It is a permutation inside each 32-vector/128-float span.
__device__ __forceinline__ int activation_slot(int vector) {
    return vector ^ ((vector >> 3) & 3);
}

template<int M, int ROWS, int SPLIT, bool WIDE>
__global__ void gemv_shared_activation(const float* __restrict__ x,
                                      const uint8_t* __restrict__ w,
                                      const uint8_t* __restrict__ scales,
                                      uint16_t* __restrict__ y, int64_t k, int64_t n) {
    static_assert(SPLIT == 1 || SPLIT == 2, "bounded tile uses one or two K warps");
    constexpr int GROUPS = THREADS / (32 * SPLIT);
    constexpr int VALUES = ACT_TILE / (32 * SPLIT);
    constexpr int WORDS = VALUES / 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int split = warp % SPLIT;
    const int group = warp / SPLIT;
    __shared__ float4 activation[M][ACT_TILE / 4];
    __shared__ float partial[THREADS / 32][ROWS][M];
    for (int64_t base = int64_t(blockIdx.x) * GROUPS * ROWS;
         base < n; base += int64_t(gridDim.x) * GROUPS * ROWS) {
        const int64_t row = base + group * ROWS;
        const int local_col = (split * 32 + lane) * VALUES;
        float acc[ROWS][M] = {};
        for (int64_t tile = 0; tile < k; tile += ACT_TILE) {
            // Each thread copies one vector per token; the final K tile is guarded.
            // No [M,K] workspace or host-side state is needed, including in capture.
            const int vector = threadIdx.x;
            const int64_t load_col = tile + vector * 4;
            if (load_col < k) {
                #pragma unroll
                for (int t = 0; t < M; ++t) {
                    const float* p = x + int64_t(t) * k + load_col;
                    float4 a;
                    if constexpr (WIDE) a = *reinterpret_cast<const float4*>(p);
                    else a = make_float4(p[0], p[1], p[2], p[3]);
                    activation[t][activation_slot(vector)] = a;
                }
            }
            __syncthreads();
            const int64_t col = tile + local_col;
            if (col < k) {
                uint4 packed[ROWS];
                const float sw = row < n ?
                    detail::decode_e8m0(scales[(row / 32) * (k / 32) + col / 32]) : 0;
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    packed[r] = make_uint4(0, 0, 0, 0);
                    if (row + r < n) {
                        const uint8_t* p = w + (row + r) * k + col;
                        if constexpr (WIDE && SPLIT == 1) {
                            packed[r] = *reinterpret_cast<const uint4*>(p);
                        } else if constexpr (WIDE) {
                            const uint2 v = *reinterpret_cast<const uint2*>(p);
                            packed[r] = make_uint4(v.x, v.y, 0, 0);
                        } else {
                            uint32_t words[4] = {};
                            #pragma unroll
                            for (int j = 0; j < VALUES; ++j)
                                words[j / 4] |= uint32_t(p[j]) << ((j % 4) * 8);
                            packed[r] = make_uint4(words[0], words[1], words[2], words[3]);
                        }
                    }
                }
                #pragma unroll
                for (int j = 0; j < WORDS; ++j) {
                    float4 weight[ROWS];
                    #pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        const uint32_t word = j == 0 ? packed[r].x : j == 1 ? packed[r].y :
                                              j == 2 ? packed[r].z : packed[r].w;
                        weight[r] = make_float4(detail::decode_e4m3(uint8_t(word)) * sw,
                                                detail::decode_e4m3(uint8_t(word >> 8)) * sw,
                                                detail::decode_e4m3(uint8_t(word >> 16)) * sw,
                                                detail::decode_e4m3(uint8_t(word >> 24)) * sw);
                    }
                    const int slot = activation_slot(local_col / 4 + j);
                    #pragma unroll
                    for (int t = 0; t < M; ++t) {
                        const float4 a = activation[t][slot];
                        #pragma unroll
                        for (int r = 0; r < ROWS; ++r) {
                            acc[r][t] = fmaf(a.x, weight[r].x, acc[r][t]);
                            acc[r][t] = fmaf(a.y, weight[r].y, acc[r][t]);
                            acc[r][t] = fmaf(a.z, weight[r].z, acc[r][t]);
                            acc[r][t] = fmaf(a.w, weight[r].w, acc[r][t]);
                        }
                    }
                }
            }
            // Every warp must finish the current tile before producers overwrite it.
            __syncthreads();
        }
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            #pragma unroll
            for (int t = 0; t < M; ++t) {
                for (int d = 16; d; d >>= 1)
                    acc[r][t] += __shfl_down_sync(0xffffffffu, acc[r][t], d);
                if (lane == 0) {
                    if constexpr (SPLIT == 1) {
                        if (row + r < n)
                            y[int64_t(t) * n + row + r] = strata::kernels::bf16_from_f32(acc[r][t]);
                    } else {
                        partial[warp][r][t] = acc[r][t];
                    }
                }
            }
        }
        if constexpr (SPLIT > 1) {
            __syncthreads();
            if (split == 0 && lane == 0) {
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    #pragma unroll
                    for (int t = 0; t < M; ++t) {
                        const float sum = partial[warp][r][t] + partial[warp + 1][r][t];
                        if (row + r < n)
                            y[int64_t(t) * n + row + r] = strata::kernels::bf16_from_f32(sum);
                    }
                }
            }
            __syncthreads();
        }
    }
}

template<int M, int SPLIT>
void launch_shared_activation(const float* x, const uint8_t* w, const uint8_t* scales,
                              uint16_t* y, int64_t k, int64_t n, cudaStream_t stream) {
    constexpr int ROWS = 2;
    constexpr int ROWS_PER_BLOCK = THREADS / (32 * SPLIT) * ROWS;
    const int64_t blocks = (n - 1) / ROWS_PER_BLOCK + 1;
    const unsigned grid = unsigned(blocks < 65535 ? blocks : 65535);
    if (((reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(x)) & 15u) == 0)
        gemv_shared_activation<M, ROWS, SPLIT, true><<<grid, THREADS, 0, stream>>>(x, w, scales, y, k, n);
    else
        gemv_shared_activation<M, ROWS, SPLIT, false><<<grid, THREADS, 0, stream>>>(x, w, scales, y, k, n);
}

unsigned grid_for(int64_t rows) {
    const int64_t grid = (rows - 1) / (THREADS / 32) + 1;
    return unsigned(grid < 65535 ? grid : 65535);
}

template<int M, int ROWS, int SPLIT>
void launch_layout(const float* x, const uint8_t* w, const uint8_t* scales, uint16_t* y,
                   int64_t k, int64_t n, cudaStream_t stream) {
    constexpr int ROWS_PER_BLOCK = (THREADS / 32 / SPLIT) * ROWS;
    const int64_t blocks = (n - 1) / ROWS_PER_BLOCK + 1;
    const unsigned grid = unsigned(blocks < 65535 ? blocks : 65535);
    if (((reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(x)) & 15u) == 0)
        gemv<M, ROWS, SPLIT, true><<<grid, THREADS, 0, stream>>>(x, w, scales, y, k, n);
    else
        gemv<M, ROWS, SPLIT, false><<<grid, THREADS, 0, stream>>>(x, w, scales, y, k, n);
}

template<int M, int ROWS>
void launch_split(const float* x, const uint8_t* w, const uint8_t* scales, uint16_t* y,
                  int64_t k, int64_t n, cudaStream_t stream) {
    switch (detail::gemv_split_warps(n)) {
        case 4: launch_layout<M, ROWS, 4>(x, w, scales, y, k, n, stream); break;
        case 2: launch_layout<M, ROWS, 2>(x, w, scales, y, k, n, stream); break;
        default: launch_layout<M, ROWS, 1>(x, w, scales, y, k, n, stream); break;
    }
}

template<int M>
void launch(const float* x, const uint8_t* w, const uint8_t* scales, uint16_t* y,
            int64_t k, int64_t n, cudaStream_t stream) {
    if constexpr (M > 1) {
        if (n > 2048) {
            if (n <= 8192) launch_shared_activation<M, 2>(x, w, scales, y, k, n, stream);
            else launch_shared_activation<M, 1>(x, w, scales, y, k, n, stream);
            return;
        }
    }
    if (detail::gemv_rows_per_group(n) == 2)
        launch_split<M, 2>(x, w, scales, y, k, n, stream);
    else
        launch_split<M, 1>(x, w, scales, y, k, n, stream);
}

void validate_output(int64_t k, const uint8_t* w, const uint8_t* scales, int64_t n, uint16_t* y) {
    if (!w || !scales || !y || n <= 0 ||
        n > std::numeric_limits<int64_t>::max() / k ||
        uint64_t(n) > std::numeric_limits<size_t>::max() / (8 * sizeof(uint16_t)))
        throw std::invalid_argument("fp8_block_gemv: invalid weight/output geometry or null pointer");
}

}  // namespace

void fp8_quantize_activation(const uint16_t* x, int m, int64_t k,
                             uint8_t* xq, float* x_scale, void* stream) {
    validate(x, m, k);
    if (!xq || !x_scale) throw std::invalid_argument("fp8_quantize_activation: null output");
    const int64_t blocks = int64_t(m) * (k / 32);
    quantize<<<grid_for(blocks), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        x, blocks, nullptr, xq, x_scale);
    check(cudaGetLastError(), "quantize launch");
}

void fp8_quantize_activation_f32(const uint16_t* x, int m, int64_t k,
                                 float* x_deq, void* stream) {
    validate(x, m, k);
    if (!x_deq) throw std::invalid_argument("fp8_quantize_activation_f32: null output");
    const int64_t blocks = int64_t(m) * (k / 32);
    quantize<<<grid_for(blocks), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        x, blocks, x_deq, nullptr, nullptr);
    check(cudaGetLastError(), "quantize f32 launch");
}

void fp8_block_gemv_q(const float* x_deq, int m, int64_t k,
                      const uint8_t* w, const uint8_t* w_scale, int64_t n,
                      uint16_t* y, void* stream) {
    validate(x_deq, m, k);
    validate_output(k, w, w_scale, n, y);
    const auto s = static_cast<cudaStream_t>(stream);
    switch (m) {
        case 1: launch<1>(x_deq, w, w_scale, y, k, n, s); break;
        case 2: launch<2>(x_deq, w, w_scale, y, k, n, s); break;
        case 3: launch<3>(x_deq, w, w_scale, y, k, n, s); break;
        case 4: launch<4>(x_deq, w, w_scale, y, k, n, s); break;
        case 5: launch<5>(x_deq, w, w_scale, y, k, n, s); break;
        case 6: launch<6>(x_deq, w, w_scale, y, k, n, s); break;
        case 7: launch<7>(x_deq, w, w_scale, y, k, n, s); break;
        case 8: launch<8>(x_deq, w, w_scale, y, k, n, s); break;
    }
    check(cudaGetLastError(), "gemv q launch");
}

void fp8_block_gemv(const uint16_t* x, int m, int64_t k,
                    const uint8_t* w, const uint8_t* w_scale, int64_t n,
                    uint16_t* y, void* stream) {
    validate(x, m, k);
    validate_output(k, w, w_scale, n, y);
    const auto s = static_cast<cudaStream_t>(stream);
    float* dequant = nullptr;
    check(cudaMallocAsync(reinterpret_cast<void**>(&dequant), size_t(m) * size_t(k) * sizeof(float), s),
          "activation allocation");
    fp8_quantize_activation_f32(x, m, k, dequant, stream);
    fp8_block_gemv_q(dequant, m, k, w, w_scale, n, y, stream);
    check(cudaFreeAsync(dequant, s), "activation release");
}

}  // namespace strata::ds41
