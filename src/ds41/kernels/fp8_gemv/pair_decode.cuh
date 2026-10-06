// Small-N single-token path. Included inside fp8_gemv.cu's anonymous namespace.
// FP16 is only an exact conversion intermediate; all scaled weights and sums are FP32.
template<bool SCALE_IN_RANGE>
__host__ __device__ __forceinline__ float2 decode_small_pair(uint16_t q, uint8_t scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(q, __NV_E4M3);
    const float2 v = __half22float2(__half2(h));
    const float sw = detail::decode_e8m0(scale);
    return make_float2(v.x * sw, v.y * sw);
#else
    // Before native FP8 conversion, expand both bytes to FP16(value / 256).
    // Its subnormals exactly represent every E4M3 subnormal, including signed zero.
    // Fold the compensating power of two into the scale, except where it overflows.
    if constexpr (!SCALE_IN_RANGE)
        return make_float2(detail::decode_e4m3(uint8_t(q)) * detail::decode_e8m0(scale),
                           detail::decode_e4m3(uint8_t(q >> 8)) * detail::decode_e8m0(scale));
#if defined(__CUDA_ARCH__)
    const uint32_t spread = __byte_perm(uint32_t(q), 0, 0x4140);
#else
    const uint32_t spread = uint32_t(q & 255u) | (uint32_t(q >> 8) << 16);
#endif
    const uint32_t bits = ((spread & 0x007f007fu) << 7) | ((spread & 0x00800080u) << 8);
    __half2_raw h;
    memcpy(&h, &bits, sizeof(bits));
    const float2 v = __half22float2(__half2(h));
    const float sw256 = detail::from_bits(uint32_t(scale + 8u) << 23);
    return make_float2((q & 0x007fu) == 0x007fu ? detail::from_bits(0x7fc00000u | (uint32_t(q & 0x80u) << 24)) : v.x * sw256,
                       (q & 0x7f00u) == 0x7f00u ? detail::from_bits(0x7fc00000u | (uint32_t(q & 0x8000u) << 16)) : v.y * sw256);
#endif
}

