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

void validate(const uint16_t* x, int m, int64_t k) {
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

// Each warp owns a row; consecutive lanes read consecutive 16-byte weight vectors.
// Decode once and keep M accumulators so verify windows do not reread the weight matrix.
template<int M, bool WIDE>
__global__ void gemv(const float* __restrict__ x, const uint8_t* __restrict__ w,
                     const uint8_t* __restrict__ scales, uint16_t* __restrict__ y,
                     int64_t k, int64_t n) {
    const int lane = threadIdx.x & 31;
    for (int64_t row = int64_t(blockIdx.x) * (THREADS / 32) + threadIdx.x / 32;
         row < n; row += int64_t(gridDim.x) * (THREADS / 32)) {
        float acc[M] = {};
        for (int64_t col = int64_t(lane) * 16; col < k; col += 512) {
            uint4 packed;
            const uint8_t* p = w + row * k + col;
            if constexpr (WIDE) {
                packed = *reinterpret_cast<const uint4*>(p);
            } else {
                // The public interface does not require 16-byte alignment for a sliced byte tensor.
                uint32_t words[4] = {};
                #pragma unroll
                for (int j = 0; j < 16; ++j) words[j / 4] |= uint32_t(p[j]) << ((j % 4) * 8);
                packed = make_uint4(words[0], words[1], words[2], words[3]);
            }
            const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            const float sw = detail::decode_e8m0(scales[(row / 32) * (k / 32) + col / 32]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float w0 = detail::decode_e4m3(uint8_t(words[j])) * sw;
                const float w1 = detail::decode_e4m3(uint8_t(words[j] >> 8)) * sw;
                const float w2 = detail::decode_e4m3(uint8_t(words[j] >> 16)) * sw;
                const float w3 = detail::decode_e4m3(uint8_t(words[j] >> 24)) * sw;
                #pragma unroll
                for (int t = 0; t < M; ++t) {
                    const float4 a = *reinterpret_cast<const float4*>(x + int64_t(t) * k + col + j * 4);
                    acc[t] = fmaf(a.x, w0, acc[t]);
                    acc[t] = fmaf(a.y, w1, acc[t]);
                    acc[t] = fmaf(a.z, w2, acc[t]);
                    acc[t] = fmaf(a.w, w3, acc[t]);
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < M; ++t) {
            for (int d = 16; d; d >>= 1) acc[t] += __shfl_down_sync(0xffffffffu, acc[t], d);
            if (lane == 0) y[int64_t(t) * n + row] = strata::kernels::bf16_from_f32(acc[t]);
        }
    }
}

unsigned grid_for(int64_t rows) {
    const int64_t grid = (rows - 1) / (THREADS / 32) + 1;
    return unsigned(grid < 65535 ? grid : 65535);
}

template<int M>
void launch(const float* x, const uint8_t* w, const uint8_t* scales, uint16_t* y,
            int64_t k, int64_t n, cudaStream_t stream) {
    if ((reinterpret_cast<uintptr_t>(w) & 15u) == 0)
        gemv<M, true><<<grid_for(n), THREADS, 0, stream>>>(x, w, scales, y, k, n);
    else
        gemv<M, false><<<grid_for(n), THREADS, 0, stream>>>(x, w, scales, y, k, n);
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

void fp8_block_gemv(const uint16_t* x, int m, int64_t k,
                    const uint8_t* w, const uint8_t* w_scale, int64_t n,
                    uint16_t* y, void* stream) {
    validate(x, m, k);
    if (!w || !w_scale || !y || n <= 0 ||
        n > std::numeric_limits<int64_t>::max() / k ||
        uint64_t(n) > std::numeric_limits<size_t>::max() / (8 * sizeof(uint16_t)))
        throw std::invalid_argument("fp8_block_gemv: invalid weight/output geometry or null pointer");
    const auto s = static_cast<cudaStream_t>(stream);
    float* dequant = nullptr;
    check(cudaMallocAsync(reinterpret_cast<void**>(&dequant), size_t(m) * size_t(k) * sizeof(float), s),
          "activation allocation");
    const int64_t blocks = int64_t(m) * (k / 32);
    quantize<<<grid_for(blocks), THREADS, 0, s>>>(x, blocks, dequant, nullptr, nullptr);
    check(cudaGetLastError(), "quantize launch");
    switch (m) {
        case 1: launch<1>(dequant, w, w_scale, y, k, n, s); break;
        case 2: launch<2>(dequant, w, w_scale, y, k, n, s); break;
        case 3: launch<3>(dequant, w, w_scale, y, k, n, s); break;
        case 4: launch<4>(dequant, w, w_scale, y, k, n, s); break;
        case 5: launch<5>(dequant, w, w_scale, y, k, n, s); break;
        case 6: launch<6>(dequant, w, w_scale, y, k, n, s); break;
        case 7: launch<7>(dequant, w, w_scale, y, k, n, s); break;
        case 8: launch<8>(dequant, w, w_scale, y, k, n, s); break;
    }
    check(cudaGetLastError(), "gemv launch");
    check(cudaFreeAsync(dequant, s), "activation release");
}

}  // namespace strata::ds41
