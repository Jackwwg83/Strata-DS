// CPU-only arithmetic model for the staged kernel, including its FP32 FMA and warp reduction.
// g++ -O2 -std=c++17 -Iinclude src/ds41/kernels/fp8_gemv/async_weights_numerics.cpp -o /tmp/k1c06-model
#include "strata/ds41/fp8_gemv.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace d = strata::ds41::detail;

float bf16(float f) {
    const uint32_t bits = d::float_bits(f);
    return d::from_bits((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
}

int main() {
    std::mt19937 rng(601024);
    std::normal_distribution<float> normal(0.f, 1.f);
    int cases = 0;
    double worst = 0;
    for (int n : {1, 3, 5, 31, 33, 63}) {
        for (int k : {32, 96, 512, 544, 1056, 2304, 3104}) {
            std::vector<uint8_t> w(n * k), scales(((n + 31) / 32) * (k / 32));
            for (auto& q : w) {
                q = uint8_t(rng() % 256);
                if ((q & 127u) == 127u) --q;
            }
            for (auto& q : scales) q = uint8_t(118 + rng() % 12);
            for (int m = 1; m <= 8; ++m) {
                std::vector<float> x(m * k);
                for (auto& f : x) f = bf16(normal(rng));
                for (int b = 0; b < m * k; b += 32) {
                    if ((b / 32) % 13 == 0) std::fill(x.begin() + b, x.begin() + b + 32, 0.f);
                    if ((b / 32) % 19 == 0) x[b] = -512.f;
                    float amax = 1e-4f;
                    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b + j]));
                    const float s = d::activation_scale(amax);
                    for (int j = 0; j < 32; ++j) x[b + j] = d::decode_e4m3(d::encode_e4m3(x[b + j] / s)) * s;
                }
                double delta = 0, norm = 0;
                for (int row = 0; row < n; ++row) {
                    std::array<std::array<float, 32>, 8> acc{};
                    for (int tile = 0; tile < k; tile += 1024) {
                        for (int lane = 0; lane < 32; ++lane) {
                            for (int v = 0; v < 2; ++v) {
                                const int col = tile + (lane + 32 * v) * 16;
                                if (col >= k) continue;
                                const float sw = d::decode_e8m0(scales[(row / 32) * (k / 32) + col / 32]);
                                for (int j = 0; j < 16; ++j) {
                                    const float weight = d::decode_e4m3(w[row * k + col + j]) * sw;
                                    for (int t = 0; t < m; ++t)
                                        acc[t][lane] = std::fma(x[t * k + col + j], weight, acc[t][lane]);
                                }
                            }
                        }
                    }
                    for (int t = 0; t < m; ++t) {
                        for (int distance = 16; distance; distance >>= 1) {
                            const auto previous = acc[t];
                            for (int lane = 0; lane + distance < 32; ++lane)
                                acc[t][lane] = previous[lane] + previous[lane + distance];
                        }
                        double ref = 0;
                        for (int col = 0; col < k; ++col) {
                            const float weight = d::decode_e4m3(w[row * k + col]) *
                                d::decode_e8m0(scales[(row / 32) * (k / 32) + col / 32]);
                            ref += double(x[t * k + col]) * weight;
                        }
                        const double expected = bf16(float(ref));
                        const double diff = double(bf16(acc[t][0])) - expected;
                        delta += diff * diff;
                        norm += expected * expected;
                    }
                }
                const double rel = std::sqrt(delta / std::max(norm, 1e-30));
                if (!(rel <= 2e-3)) {
                    std::printf("FAIL n=%d k=%d m=%d rel=%g\n", n, k, m, rel);
                    return 1;
                }
                worst = std::max(worst, rel);
                ++cases;
            }
        }
    }
    std::printf("PASS %d CPU FP32/BF16 cases, worst relative L2=%g; GPU execution is untested\n", cases, worst);
}
