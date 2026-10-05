// Exact E4M3 * E8M0 -> BF16 conversion, also host-callable for exhaustive tests.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstring>

namespace strata::ds41::kernels::k2_detail {

__host__ __device__ __forceinline__ bool normal_four(unsigned packed, unsigned scale_byte) {
    // Each addition stays within its byte: exponent != 0 sets bit 7 in normal,
    // while magnitude == 127 (the two E4M3 NaNs) sets bit 7 in nan.
    const unsigned normal = (packed & 0x78787878u) + 0x78787878u;
    const unsigned nan = (packed & 0x7f7f7f7fu) + 0x01010101u;
    // E4M3 exponent e becomes BF16 exponent e + scale_byte - 7.
    // This range keeps every e=1..15 normal and finite, with no rounding.
    return scale_byte - 7u <= 239u &&
           ((normal & ~nan) & 0x80808080u) == 0x80808080u;
}

__host__ __device__ __forceinline__ unsigned widen_pair(unsigned packed) {
#if defined(__CUDA_ARCH__)
    return __byte_perm(packed, 0u, 0x4140);
#else
    return (packed & 0xffu) | ((packed & 0xff00u) << 8);
#endif
}

__host__ __device__ __forceinline__ unsigned pair_bits(unsigned spaced, unsigned bias) {
    // Two independent 16-bit lanes. The finite normal range prevents either
    // magnitude addition from carrying into the sign bit or the other lane.
    return (((spaced & 0x007f007fu) << 4) + bias) |
           ((spaced & 0x00800080u) << 8);
}

__host__ __device__ __forceinline__ void store_four(
    __nv_bfloat16* dst, unsigned packed, unsigned scale_byte) {
    if (normal_four(packed, scale_byte)) {
        const unsigned bias = (scale_byte - 7u) * 0x00800080u;
        const unsigned lo = pair_bits(widen_pair(packed), bias);
        const unsigned hi = pair_bits(widen_pair(packed >> 16), bias);
#if defined(__CUDA_ARCH__)
        const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
                     :: "r"(address), "r"(lo), "r"(hi) : "memory");
#else
        // memcpy preserves the bit representation without host strict-aliasing
        // violations; the device stores the same two words directly to shared.
        std::memcpy(dst, &lo, sizeof(lo));
        std::memcpy(dst + 2, &hi, sizeof(hi));
#endif
        return;
    }
    // Keep the original CUDA conversion for FP8 subnormals, signed zero,
    // NaNs, BF16 under/overflow and E8M0 byte 255 (FP32 infinity).
    const float scale = ldexpf(1.0f, static_cast<int>(scale_byte) - 127);
    __nv_fp8x4_e4m3 q;
    q.__x = packed;
    const float4 value = static_cast<float4>(q);
    auto* pair = reinterpret_cast<__nv_bfloat162*>(dst);
    pair[0] = __floats2bfloat162_rn(value.x * scale, value.y * scale);
    pair[1] = __floats2bfloat162_rn(value.z * scale, value.w * scale);
}

}  // namespace strata::ds41::kernels::k2_detail
