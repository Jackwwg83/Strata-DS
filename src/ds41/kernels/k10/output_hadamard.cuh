#pragma once

// Copyright (c) Turboderp. MIT license; see third_party/exllamav3_gpu/LICENSE.

// Return-only adaptation of exllamav3 16a49792's had_ff_r_128_inner.
// The load, butterflies, shuffle order and separate scale multiplies are exact
// upstream source. Only the FP32 store becomes a register return. The unchanged
// upstream helper remains the independent reference for host checks.
// Requires the upstream hadamard_inner.cuh (included by the unity GEMV unit).
namespace strata::ds41::kernels::k10 {

template <bool pre_scale, bool post_scale>
inline __device__
float4 had_ff_r_128_registers
(
    const float* __restrict__ input_ptr,
    const half* __restrict__ scale,
    const float r_scale
)
{
    int t = threadIdx.x & 31;

    // Load
    float4 v = ((float4*) input_ptr)[t];

    // Pre scale
    if constexpr (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    // 4 element had
    float v0 = v.x;
    float v1 = v.y;
    float v2 = v.z;
    float v3 = v.w;
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;

    // 32 element had, warp shuffle
    shuffle_had_f2x32(v.x, v.y, t);
    shuffle_had_f2x32(v.z, v.w, t);
    v.x *= r_scale;
    v.y *= r_scale;
    v.z *= r_scale;
    v.w *= r_scale;

    // Post scale
    if constexpr (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    // The caller owns the same lane*4 output coordinates.
    return v;
}

}  // namespace strata::ds41::kernels::k10
