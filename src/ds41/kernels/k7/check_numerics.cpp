// CPU arithmetic model only. CUDA kernel execution/libm remain untested.
#include "register_prefetch.hpp"
#include <algorithm>
#include <cassert>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

constexpr int N = 20480, Rows = 24, DotThreads = 256, NormThreads = 1024;
using Coeff = std::array<float, Rows>;

uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
float bf16(float value) {
    uint32_t u = bits(value);
    u += 0x7fff + ((u >> 16) & 1);
    u &= 0xffff0000;
    std::memcpy(&value, &u, sizeof(value));
    return value;
}
std::vector<float> randoms(int n, float scale, int seed, bool bf = false) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0, scale);
    std::vector<float> values(n);
    for (float& value : values) value = bf ? bf16(distribution(generator)) : distribution(generator);
    return values;
}
float warp_sum(std::array<float, 32> values) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const auto old = values;
        for (int lane = 0; lane < 32; ++lane)
            values[lane] += old[lane + offset < 32 ? lane + offset : lane];
    }
    return values[0];
}
float reference_sum(const float* x, const float* weights, bool norm) {
    const int threads = norm ? NormThreads : DotThreads;
    float sum = 0;
    for (int warp = 0; warp < threads / 32; ++warp) {
        std::array<float, 32> values{};
        for (int lane = 0; lane < 32; ++lane)
            for (int column = warp * 32 + lane; column < N; column += threads)
                values[lane] = std::fma(x[column], norm ? x[column] : weights[column], values[lane]);
        sum += warp_sum(values);
    }
    return sum;
}
struct Raw {
    float dots[8][Rows][8] = {};
    float squares[8][32] = {};
};

template <int Tokens>
struct CpuLoad {
    const float* x;
    const float* weights;
    int original_lane;
    int* visits;
    strata::ds41::kernels::k7_detail::Stage<Tokens> operator()(int step) const {
        assert(step >= 0 && step < 80);
        ++visits[step];
        const int column = original_lane + step * DotThreads;
        assert(column >= 0 && column < N);
        strata::ds41::kernels::k7_detail::Stage<Tokens> stage;
        stage.weight = weights[column];
        for (int token = 0; token < Tokens; ++token)
            stage.values[token] = x[token * N + column];
        return stage;
    }
};
struct CpuFma {
    float operator()(float a, float b, float c) const { return std::fma(a, b, c); }
};
template <int Tokens>
Raw candidate(const float* x, const float* weights) {
    Raw result;
    for (int row = 0; row < Rows; ++row)
        for (int warp = 0; warp < 8; ++warp) {
            std::array<float, 32> dots[Tokens], squares[Tokens];
            for (int lane = 0; lane < 32; ++lane) {
                float d[Tokens] = {}, s[Tokens] = {};
                int visits[80] = {};
                strata::ds41::kernels::k7_detail::register_prefetch<Tokens, 80>(
                    CpuLoad<Tokens>{x, weights + row * N, warp * 32 + lane, visits},
                    CpuFma{}, row, d, s);
                for (int n : visits) assert(n == 1);
                for (int token = 0; token < Tokens; ++token) {
                    dots[token][lane] = d[token];
                    squares[token][lane] = s[token];
                }
            }
            for (int token = 0; token < Tokens; ++token) {
                result.dots[token][row][warp] = warp_sum(dots[token]);
                if (row < 4) result.squares[token][row * 8 + warp] = warp_sum(squares[token]);
            }
        }
    return result;
}
Raw dispatch(int m, const float* x, const float* weights) {
    switch (m) {
#define CASE(M) case M: return candidate<M>(x, weights)
        CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8);
#undef CASE
    }
    std::abort();
}
float rejected_tile_sum(const float* x, const float* weights) {
    float sum = 0;
    for (int part = 0; part < 20; ++part) {
        float block = 0;
        for (int warp = 0; warp < 8; ++warp) {
            std::array<float, 32> values{};
            for (int lane = 0; lane < 32; ++lane)
                for (int offset = 0; offset < 1024; offset += 256) {
                    const int column = part * 1024 + offset + warp * 32 + lane;
                    values[lane] = std::fma(x[column], weights[column], values[lane]);
                }
            block += warp_sum(values);
        }
        sum += block;
    }
    return sum;
}
Coeff scalar_finish(const Coeff& mix, float reciprocal_rms, const std::vector<float>& base) {
    const float scale[3] = {0.7f, 0.9f, 1.3f};
    Coeff result;
    for (int j = 0; j < 4; ++j) {
        result[j] = 1.f / (1.f + std::exp(-std::fma(mix[j] * reciprocal_rms, scale[0], base[j]))) + 1e-6f;
        result[j + 4] = 2.f / (1.f + std::exp(-std::fma(mix[j + 4] * reciprocal_rms, scale[1], base[j + 4])));
    }
    float c[4][4];
    for (int row = 0; row < 4; ++row) {
        float maximum = -INFINITY;
        for (int column = 0; column < 4; ++column) {
            const int index = 8 + row * 4 + column;
            c[row][column] = std::fma(mix[index] * reciprocal_rms, scale[2], base[index]);
            maximum = std::fmax(maximum, c[row][column]);
        }
        float sum = 0;
        for (int column = 0; column < 4; ++column) {
            c[row][column] = std::exp(c[row][column] - maximum);
            sum += c[row][column];
        }
        for (int column = 0; column < 4; ++column) c[row][column] = c[row][column] / sum + 1e-6f;
    }
    auto normalize_columns = [&] {
        for (int column = 0; column < 4; ++column) {
            float sum = 0;
            for (int row = 0; row < 4; ++row) sum += c[row][column];
            for (int row = 0; row < 4; ++row) c[row][column] /= sum + 1e-6f;
        }
    };
    normalize_columns();
    for (int iteration = 0; iteration < 19; ++iteration) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0;
            for (int column = 0; column < 4; ++column) sum += c[row][column];
            for (int column = 0; column < 4; ++column) c[row][column] /= sum + 1e-6f;
        }
        normalize_columns();
    }
    for (int index = 0; index < 16; ++index) result[8 + index] = c[index / 4][index % 4];
    return result;
}
Coeff warp_finish(const Coeff& mix, float reciprocal_rms, const std::vector<float>& base) {
    Coeff result = scalar_finish(mix, reciprocal_rms, base);
    std::array<float, 32> c, next;
    for (int lane = 0; lane < 32; ++lane) {
        const int index = 8 + (lane & 15);
        c[lane] = std::fma(mix[index] * reciprocal_rms, 1.3f, base[index]);
    }
    for (int lane = 0; lane < 32; ++lane) {
        float maximum = -INFINITY;
        for (int k = 0; k < 4; ++k) maximum = std::fmax(maximum, c[(lane & 12) + k]);
        next[lane] = std::exp(c[lane] - maximum);
    }
    c = next;
    for (int lane = 0; lane < 32; ++lane) {
        float sum = 0;
        for (int k = 0; k < 4; ++k) sum += c[(lane & 12) + k];
        next[lane] = c[lane] / sum + 1e-6f;
    }
    c = next;
    auto normalize = [&](bool column) {
        for (int lane = 0; lane < 32; ++lane) {
            float sum = 0;
            for (int k = 0; k < 4; ++k) sum += c[column ? k * 4 + (lane & 3) : (lane & 12) + k];
            next[lane] = c[lane] / (sum + 1e-6f);
        }
        c = next;
    };
    normalize(true);
    for (int iteration = 0; iteration < 19; ++iteration) {
        normalize(false);
        normalize(true);
    }
    for (int lane = 0; lane < 16; ++lane) result[8 + lane] = c[lane];
    return result;
}
int main() {
    const auto base = randoms(Rows, 0.5f, 2);
    int tokens = 0, cases = 0;
    for (int scenario = 0; scenario < 12; ++scenario) {
        auto weights = randoms(Rows * N, 1 / std::sqrt(float(N)), 1103 + scenario * 31);
        for (int m = 1; m <= 8; ++m) {
            auto x = randoms(m * N, 2.f, 2027 + m + scenario * 97, true);
            if (scenario == 1) std::fill(x.begin(), x.end(), 0.f);
            if (scenario == 2)
                for (int i = 0; i < int(x.size()); ++i) x[i] = (i % 2) ? -1.f : 1.f;
            if (scenario == 3)
                for (int i = 0; i < int(x.size()); ++i) x[i] = (i % 1024 == 1023) ? 100.f : 0.f;
            if (scenario == 4)
                for (float& v : x) v = bf16(v * 1e-8f);
            if (scenario == 5) {
                // Exact router-mix-review/k7_cancellation.cpp regression.
                std::fill(x.begin(), x.end(), 1.f);
                std::fill(weights.begin(), weights.end(), 0.f);
                weights[0] = float(1u << 25);
                weights[1024] = -float(1u << 25);
                weights[1280] = 1.f;
            }
            if (scenario == 6) {
                for (float& v : x) v = bf16(v * 1e14f);
                for (int i = 0; i < int(weights.size()); ++i)
                    weights[i] = std::ldexp((i & 1) ? -1.0078125f : 1.f, (i % 91) - 45);
            }
            if (scenario == 7) {
                for (float& v : x) v = bf16(v * 1e-14f);
                for (int i = 0; i < int(weights.size()); ++i)
                    weights[i] = std::ldexp((i & 1) ? -1.f : 1.0078125f, (i % 131) - 30);
            }
            if (scenario == 8) {
                for (int i = 0; i < int(x.size()); ++i)
                    x[i] = std::ldexp((i & 1) ? -1.0078125f : 1.f, (i * 13 % 111) - 55);
                for (int i = 0; i < int(weights.size()); ++i)
                    weights[i] = std::ldexp((i & 2) ? -1.00390625f : 1.f, (i * 17 % 111) - 55);
            }
            if (scenario == 9)
                for (float& v : x) v = bf16(v * std::ldexp(1.f, -100));
            if (scenario == 10) {
                for (int i = 0; i < int(x.size()); ++i)
                    x[i] = i % 2 ? -0.f : std::ldexp(1.f, -130);
                for (int i = 0; i < int(weights.size()); ++i)
                    weights[i] = std::ldexp((i & 1) ? -1.f : 1.f, (i % 91) - 140);
            }
            if (scenario == 11) {
                // Move cancellation to every row/lane and pipeline-pair boundary.
                std::fill(x.begin(), x.end(), 1.f);
                std::fill(weights.begin(), weights.end(), 0.f);
                for (int row = 0; row < Rows; ++row) {
                    const int lane = row * 11 % 256;
                    const int step = 2 * (row % 38);
                    weights[row * N + lane + step * 256] = float(1u << 25);
                    weights[row * N + lane + (step + 1) * 256] = -float(1u << 25);
                    weights[row * N + lane + (step + 2) * 256] = 1.f;
                }
            }
            const Raw raw = dispatch(m, x.data(), weights.data());
            for (int token = 0; token < m; ++token) {
                const float* xt = x.data() + token * N;
                Coeff reference, candidate;
                for (int row = 0; row < Rows; ++row) {
                    reference[row] = reference_sum(xt, weights.data() + row * N, false);
                    candidate[row] = 0.f;
                    for (int warp = 0; warp < 8; ++warp) candidate[row] += raw.dots[token][row][warp];
                    if (bits(reference[row]) != bits(candidate[row])) {
                        std::printf("FAIL raw dot: scenario=%d m=%d token=%d row=%d ref=%.9g candidate=%.9g\n",
                                    scenario, m, token, row, reference[row], candidate[row]);
                        return 1;
                    }
                }
                const float norm_ref = reference_sum(xt, nullptr, true);
                float norm_new = 0.f;
                for (int warp = 0; warp < 32; ++warp) norm_new += raw.squares[token][warp];
                if (bits(norm_ref) != bits(norm_new)) return 2;
                const float reciprocal_rms = 1 / std::sqrt(norm_ref / float(N) + 1e-20f);
                const auto a = scalar_finish(reference, reciprocal_rms, base);
                const auto b = warp_finish(candidate, reciprocal_rms, base);
                if (std::memcmp(a.data(), b.data(), sizeof(a)) != 0) return 3;
                if (scenario == 5) {
                    if (reference[0] != 1.f || candidate[0] != 1.f ||
                        rejected_tile_sum(xt, weights.data()) != 0.f) return 4;
                    const float ref_pre = 1.f / (1.f + std::exp(-reference[0])) + 1e-6f;
                    const float new_pre = 1.f / (1.f + std::exp(-candidate[0])) + 1e-6f;
                    if (bits(ref_pre) != bits(new_pre) || std::fabs(ref_pre - 0.7310596f) > 1e-7f) return 5;
                }
                if (scenario == 11)
                    for (float dot : candidate) if (dot != 1.f) return 6;
                ++tokens;
            }
            ++cases;
        }
    }
    std::printf("PASS %d CPU batched cases, %d token cases, all m=1..8: exact raw dot/norm and coefficient equality\n", cases, tokens);
    std::puts("PASS actual shared pipeline body: all 80 steps loaded exactly once; bounds checked");
    std::puts("PASS original cancellation: reference=1, K7-06=1, rejected contiguous tiles=0");
    std::puts("PASS wide dynamic range, subnormal values, sparse, zero, sign and pair-boundary cases");
    std::puts("CPU arithmetic only. CUDA parity, graph replay, sanitizer and timing remain untested.");
}
