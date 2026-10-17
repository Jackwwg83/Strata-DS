// src/ds41/tests/mxfp4_ref.hpp - the CPU reference of one MXFP4 routed expert (K16's contract), for the GPU and CPU
// kernel tests: DeepSeek's Expert.forward with fp4_gemm, written out plainly.
#pragma once

#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ds41test {

inline float e2m1(int code) {
    static const float t[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return t[code & 15];
}
inline float e8m0(uint8_t e) { return std::ldexp(1.0f, (int) e - 127); }
/// round to BF16, nearest even
inline float bf16r(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&f, &u, 4);
    return f;
}
/// K10's power-of-2 scale: amax rounded up to a power of two
inline float round_pow2(float a) {
    uint32_t u;
    std::memcpy(&u, &a, 4);
    u = (u + 0x007fffffu) & 0x7f800000u;
    std::memcpy(&a, &u, 4);
    return a;
}

/// one projection: y[r] = sum over 32-blocks b of scale[r][b] * sum_i E2M1(code[r][32b + i]) * x[32b + i]
inline void mxfp4_proj(const uint8_t* codes, const uint8_t* scales, int n, int k, const float* x, float* y) {
    for (int r = 0; r < n; ++r) {
        double acc = 0;
        for (int b = 0; b < k / 32; ++b) {
            double s = 0;
            for (int i = 0; i < 32; ++i) {
                const uint8_t byte = codes[(size_t) r * (k / 2) + 16 * b + i / 2];
                s += (double) e2m1(i % 2 ? byte >> 4 : byte & 15) * x[32 * b + i];
            }
            acc += s * e8m0(scales[(size_t) r * (k / 32) + b]);
        }
        y[r] = (float) acc;
    }
}

struct Mxfp4ExpertHost {
    std::vector<uint8_t> w1c, w1s, w3c, w3s, w2c, w2s;   ///< codes and scales of gate, up, down
};

/// y = bf16(W2 qdq(bf16(silu(min(g, 10)) * clamp(u, 10) * weight))), g = bf16(W1 x), u = bf16(W3 x); x [dim], y [dim]
inline void mxfp4_expert_ref(const Mxfp4ExpertHost& e, const float* x, float weight, int dim, int inter, float* y) {
    std::vector<float> g(inter), u(inter), h(inter);
    mxfp4_proj(e.w1c.data(), e.w1s.data(), inter, dim, x, g.data());
    mxfp4_proj(e.w3c.data(), e.w3s.data(), inter, dim, x, u.data());
    for (int i = 0; i < inter; ++i) {
        const float gg = std::fmin(bf16r(g[i]), 10.0f);
        const float uu = std::fmin(std::fmax(bf16r(u[i]), -10.0f), 10.0f);
        h[i] = bf16r((gg / (1.0f + std::exp(-gg))) * uu * weight);
    }
    for (int b = 0; b < inter / 32; ++b) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(h[32 * b + i]));
        const float scale = round_pow2(std::fmax(amax, 1e-4f) * (1.0f / 448.0f));
        for (int i = 0; i < 32; ++i) {
            const __nv_fp8_e4m3 q(std::fmin(std::fmax(h[32 * b + i] / scale, -448.0f), 448.0f));
            h[32 * b + i] = bf16r(float(q) * scale);
        }
    }
    mxfp4_proj(e.w2c.data(), e.w2s.data(), dim, inter, h.data(), y);
    for (int r = 0; r < dim; ++r) y[r] = bf16r(y[r]);
}

}  // namespace ds41test
