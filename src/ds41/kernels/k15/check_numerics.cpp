// CPU arithmetic model only. CUDA kernel execution/libm remain untested.
#include "exact_accumulate.hpp"
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
struct CpuLoad {
    const float* x;
    const float* weights;
    int tid;
    int* visits;
    float value(int step) const {
        assert(step >= 0 && step < 80);
        ++visits[step];
        return x[tid + step * DotThreads];
    }
    float weight(int row, int step) const {
        assert(row >= 0 && row < Rows && step >= 0 && step < 80);
        return weights[row * N + tid + step * DotThreads];
    }
};
struct CpuFma {
    float operator()(float a, float b, float c) const { return std::fma(a, b, c); }
};
struct Raw {
    Coeff dots{};
    float norm = 0.f;
};
Raw candidate_raw(const float* x, const float* weights) {
    std::array<float, 32> dots[Rows][8], squares[32];
    for (int tid = 0; tid < DotThreads; ++tid) {
        float d[Rows] = {}, s[4] = {};
        int visits[80] = {};
        strata::ds41::kernels::k15_detail::accumulate(
            CpuLoad{x, weights, tid, visits}, CpuFma{}, d, s);
        for (int count : visits) assert(count == 1);
        for (int row = 0; row < Rows; ++row) dots[row][tid / 32][tid % 32] = d[row];
        for (int phase = 0; phase < 4; ++phase) squares[phase * 8 + tid / 32][tid % 32] = s[phase];
    }
    Raw out;
    for (int row = 0; row < Rows; ++row)
        for (int warp = 0; warp < 8; ++warp) out.dots[row] += warp_sum(dots[row][warp]);
    for (int warp = 0; warp < 32; ++warp) out.norm += warp_sum(squares[warp]);
    return out;
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
    Coeff result;
    for (int lane = 0; lane < 4; ++lane) {
        const float pm = mix[lane] * reciprocal_rms;
        const float qm = mix[lane + 4] * reciprocal_rms;
        result[lane] = 1.f / (1.f + std::exp(-std::fma(pm, 0.7f, base[lane]))) + 1e-6f;
        result[lane + 4] = 2.f / (1.f + std::exp(-std::fma(qm, 0.9f, base[lane + 4])));
    }
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
    auto weights = randoms(Rows * N, 1.f / std::sqrt(float(N)), 1);
    int checked = 0;
    auto check = [&](const float* x, const float* pin, const char* name, int token) {
        const Raw raw = candidate_raw(x, weights.data());
        Coeff reference;
        for (int row = 0; row < Rows; ++row) {
            reference[row] = reference_sum(x, weights.data() + row * N, false);
            if (bits(raw.dots[row]) != bits(reference[row])) {
                std::printf("FAIL dot %s token=%d row=%d\n", name, token, row);
                std::exit(1);
            }
        }
        const float norm = reference_sum(x, nullptr, true);
        if (bits(raw.norm) != bits(norm)) { std::puts("FAIL norm"); std::exit(2); }
        const float rms = 1.f / std::sqrt(norm / float(N) + 1e-20f);
        const auto a = scalar_finish(reference, rms, base);
        const auto b = warp_finish(raw.dots, rms, base);
        if (std::memcmp(a.data(), b.data(), sizeof(a))) {
            std::printf("FAIL coefficients %s token=%d\n", name, token); std::exit(3);
        }
        std::vector<float> y(5120);
        for (int tid = 0; tid < 256; ++tid)
            for (int d = tid; d < 5120; d += 256) {
                float sum = 0.f;
                for (int j = 0; j < 4; ++j) sum = std::fma(pin[j], x[j * 5120 + d], sum);
                y[d] = bf16(sum);
            }
        for (int d = 0; d < 5120; ++d) {
            float sum = 0.f;
            for (int j = 0; j < 4; ++j) sum = std::fma(pin[j], x[j * 5120 + d], sum);
            if (bits(y[d]) != bits(bf16(sum))) { std::puts("FAIL collapse"); std::exit(4); }
        }
        ++checked;
    };
    for (int m : {1, 37, 300}) {
        const auto x = randoms(m * N, 2.f, 10 + m, true);
        auto pin = randoms(m * 4, 0.5f, 20 + m);
        for (float& p : pin) p = std::fabs(p) + 0.01f;
        for (int token = 0; token < m; ++token)
            check(x.data() + token * N, pin.data() + token * 4, "fixed seeds", token);
        std::printf("PASS fixed seeded m=%d: every token exact raw dot/norm, coefficients and BF16 y\n", m);
    }
    for (int scenario = 0; scenario < 12; ++scenario) {
        auto x = randoms(N, 2.f, 123 + scenario, true);
        weights = randoms(Rows * N, 1.f / std::sqrt(float(N)), 113 + scenario);
        if (scenario == 0) std::fill(x.begin(), x.end(), 0.f);
        if (scenario == 1) for (int i = 0; i < N; ++i) x[i] = (i & 1) ? -1.f : 1.f;
        if (scenario == 2) for (int i = 0; i < N; ++i) x[i] = (i % 1024 == 1023) ? 100.f : 0.f;
        if (scenario == 3) for (float& v : x) v = bf16(v * 1e-8f);
        if (scenario == 4) for (float& v : x) v = bf16(v * 1e14f);
        if (scenario == 5) for (float& v : x) v = bf16(v * 1e-14f);
        if (scenario == 6) {
            for (int i = 0; i < N; ++i) x[i] = std::ldexp((i & 1) ? -1.0078125f : 1.f, (i * 13 % 111) - 55);
            for (int i = 0; i < int(weights.size()); ++i)
                weights[i] = std::ldexp((i & 2) ? -1.00390625f : 1.f, (i * 17 % 111) - 55);
        }
        if (scenario == 7) for (float& v : x) v = bf16(v * std::ldexp(1.f, -100));
        if (scenario == 8) {
            for (int i = 0; i < N; ++i) x[i] = (i & 1) ? -0.f : std::ldexp(1.f, -130);
            for (int i = 0; i < int(weights.size()); ++i)
                weights[i] = std::ldexp((i & 1) ? -1.f : 1.f, (i % 91) - 140);
        }
        if (scenario == 9 || scenario == 10) {
            std::fill(x.begin(), x.end(), 1.f);
            std::fill(weights.begin(), weights.end(), 0.f);
            for (int row = 0; row < Rows; ++row) {
                const int lane = (row * 11) % 256;
                const int step = scenario == 9 ? 2 * (row % 38) : row % 4;
                weights[row * N + lane + step * 256] = float(1u << 25);
                weights[row * N + lane + (step + 4) * 256] = -float(1u << 25);
                weights[row * N + lane + (step + 5) * 256] = 1.f;
            }
        }
        if (scenario == 11) {
            for (int i = 0; i < N; ++i) x[i] = (i < 1024) ? 1e10f : bf16((i % 13) * 1e-5f);
        }
        const float pin[4] = {0.01f, 0.25f, 0.5f, 1.f};
        check(x.data(), pin, "adversarial", scenario);
        if (scenario == 9 || scenario == 10) {
            const auto raw = candidate_raw(x.data(), weights.data());
            for (float dot : raw.dots) assert(dot == 1.f);
        }
    }
    std::printf("PASS %d token cases: bitwise host-model equality, including 12 adversarial cases\n", checked);
    std::puts("PASS shared production accumulation body: every stride-256 step loaded exactly once per lane");
    std::puts("CPU arithmetic only: GPU libm/rsqrt, numerical parity, graph replay, sanitizer and timing remain untested.");
}
