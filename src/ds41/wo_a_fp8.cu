// src/ds41/wo_a_fp8.cu - see include/strata/ds41/wo_a_fp8.hpp.
#include "strata/ds41/wo_a_fp8.hpp"

#include "strata/ds41/config.hpp"

#include <cuda_fp8.h>

#include <stdexcept>
#include <string>

namespace strata::ds41 {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kRows = kOGroups * kOLora;              // 8192
constexpr int kIn = kHeads * kHeadDim / kOGroups;     // 4096
constexpr int kBlock = 32;                            // scale block: 32 x 32
constexpr int kUnroll = 8;                            // bytes in flight per lane (RTX 4090: 4 763, 8 873, 16 823 GB/s)

/// FP8 E4M3 times an E8M0 scale: exact in float, so equal to the BF16 value convert.py stores
__device__ __forceinline__ float deq(uint8_t w, uint8_t e) {
    __nv_fp8_e4m3 v;
    v.__x = w;
    return float(v) * __uint_as_float((uint32_t) e << 23);
}

// One warp per output row, as ops::wo_a_grouped: lane i adds elements i, i + 32, ... in this order, then the same
// shuffle reduction, so the sum is the BF16 kernel's. kUnroll loads are in flight per lane.
__global__ void wo_a_fp8_k(const bf16* o, const uint8_t* w, const uint8_t* scale, bf16* y) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= kRows) return;
    const bf16* x = o + (row / kOLora) * kIn;
    const uint8_t* wr = w + (int64_t) row * kIn;
    const uint8_t* sr = scale + (row / kBlock) * (kIn / kBlock);
    float acc = 0.0f;
    for (int i = lane; i < kIn; i += kUnroll * 32) {
        uint8_t wv[kUnroll], sv[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            wv[u] = wr[i + 32 * u];
            sv[u] = sr[i / kBlock + u];
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) acc += __bfloat162float(x[i + 32 * u]) * deq(wv[u], sv[u]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = __float2bfloat16_rn(acc);
}

__global__ void dequant_k(const uint8_t* w, const uint8_t* scale, bf16* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) kRows * kIn) return;
    const int r = (int) (i / kIn), c = (int) (i % kIn);
    out[i] = __float2bfloat16_rn(deq(w[i], scale[(r / kBlock) * (kIn / kBlock) + c / kBlock]));
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 wo_a fp8: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

void wo_a_grouped_fp8(const bf16* o, const uint8_t* w, const uint8_t* scale, bf16* y, cudaStream_t stream) {
    wo_a_fp8_k<<<kRows / 8, 256, 0, stream>>>(o, w, scale, y);
    check(cudaGetLastError(), "wo_a_grouped_fp8");
}

void dequant_wo_a(const uint8_t* w, const uint8_t* scale, bf16* out, cudaStream_t stream) {
    const int64_t n = (int64_t) kRows * kIn;
    dequant_k<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(w, scale, out);
    check(cudaGetLastError(), "dequant_wo_a");
}

}  // namespace strata::ds41
