// CPU-only operation-order and ownership model. This is NOT a CUDA test.
// Build: c++ -std=c++17 -O2 -ffp-contract=off host_model.cpp -o /tmp/k7-model
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

constexpr int D = 5120, N = 4 * D, R = 24;
uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
float bf16(float x) {
    uint32_t b = bits(x);
    b = (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&x, &b, 4);
    return x;
}
float warp_sum(std::array<float, 32> a) {
    for (int off = 16; off; off >>= 1)
        for (int lane = 0; lane < off; ++lane) a[lane] += a[lane + off];
    return a[0];
}
template <int Threads>
float reduce(const std::array<float, Threads>& a) {
    float out = 0.0f;
    for (int w = 0; w < Threads / 32; ++w) {
        std::array<float, 32> warp{};
        for (int l = 0; l < 32; ++l) warp[l] = a[w * 32 + l];
        out += warp_sum(warp);
    }
    return out;
}
float reference_dot(const std::vector<float>& x, const float* f) {
    std::array<float, 256> acc{};
    for (int col = 0; col < N; ++col)
        acc[col % 256] = std::fma(x[col], f[col], acc[col % 256]);
    return reduce<256>(acc);
}
float candidate_dot(const std::vector<float>& cached, const float* f) {
    std::array<std::array<float, 32>, 8> acc{};
    for (int lane = 0; lane < 32; ++lane)
        for (int i = lane; i < N; i += 256)
            for (int w = 0; w < 8; ++w) {
                int col = i + w * 32;
                acc[w][lane] = std::fma(cached[col], f[col], acc[w][lane]);
            }
    float out = 0.0f;
    for (int w = 0; w < 8; ++w) out += warp_sum(acc[w]);
    return out;
}
std::array<float, 24> finish(const std::array<float, 24>& mix) {
    std::array<float, 24> out{};
    const float scale[3] = {0.7f, 0.9f, 1.3f};
    for (int j = 0; j < 4; ++j) {
        out[j] = 1.0f / (1.0f + std::exp(-std::fma(mix[j], scale[0], j * 0.1f))) + 1e-6f;
        out[j + 4] = 2.0f / (1.0f + std::exp(-std::fma(mix[j + 4], scale[1], j * -0.1f)));
    }
    float c[4][4];
    for (int j = 0; j < 4; ++j) {
        float mx = -INFINITY;
        for (int k = 0; k < 4; ++k) {
            c[j][k] = std::fma(mix[8 + 4 * j + k], scale[2], (4 * j + k) * 0.03f);
            mx = std::fmax(mx, c[j][k]);
        }
        float s = 0;
        for (int k = 0; k < 4; ++k) { c[j][k] = std::exp(c[j][k] - mx); s += c[j][k]; }
        for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / s + 1e-6f;
    }
    auto cols = [&] {
        for (int k = 0; k < 4; ++k) {
            float s = 0;
            for (int j = 0; j < 4; ++j) s += c[j][k];
            for (int j = 0; j < 4; ++j) c[j][k] /= s + 1e-6f;
        }
    };
    cols();
    for (int it = 0; it < 19; ++it) {
        for (int j = 0; j < 4; ++j) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += c[j][k];
            for (int k = 0; k < 4; ++k) c[j][k] /= s + 1e-6f;
        }
        cols();
    }
    for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k) out[8 + j * 4 + k] = c[j][k];
    return out;
}
void check_ownership() {
    for (int m = 1; m <= 8; ++m) {
        std::vector<int> cache(m * N), y(m * D), mix(m * R), pre(m * 4), post(m * 4), comb(m * 16);
        for (int t = 0; t < m; ++t) {
            for (int tid = 0; tid < 1024; ++tid) {
                for (int i = tid; i < N; i += 1024) ++cache[t * N + i];
                int warp = tid / 32, lane = tid % 32;
                if (warp < 24 && lane == 0) ++mix[t * R + warp];
                if (warp >= 24)
                    for (int d = tid - 768; d < D; d += 256) { assert(d >= 0); ++y[t * D + d]; }
                if (tid == 0) {
                    for (int j = 0; j < 4; ++j) { ++pre[t * 4 + j]; ++post[t * 4 + j]; }
                    for (int j = 0; j < 16; ++j) ++comb[t * 16 + j];
                }
            }
        }
        for (const auto* v : {&cache, &y, &mix, &pre, &post, &comb})
            for (int n : *v) assert(n == 1);
    }
    std::vector<int> dot(N);
    for (int lane = 0; lane < 32; ++lane)
        for (int i = lane; i < N; i += 256)
            for (int w = 0; w < 8; ++w) { int c = i + 32 * w; assert(c < N); ++dot[c]; }
    for (int n : dot) assert(n == 1);
}
int main() {
    check_ownership();
    std::mt19937 rng(94137);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> fn(R * N), x(N), cached(N);
    for (float& f : fn) f = normal(rng) / std::sqrt(float(N));
    int tokens = 0;
    for (int pattern = 0; pattern < 7; ++pattern) {
        for (int m = 1; m <= 8; ++m) {
            for (int t = 0; t < m; ++t) {
                for (int i = 0; i < N; ++i) {
                    float v = normal(rng) * 2;
                    if (pattern == 1) v = 0;
                    if (pattern == 2) v = 1;
                    if (pattern == 3) v = (i & 1) ? -32.0f : 32.0f;
                    if (pattern == 4) v *= 1e-10f;
                    if (pattern == 5) v *= 1e8f;
                    if (pattern == 6 && i % 1021) v = 0;
                    x[i] = bf16(v);
                }
                std::array<float, 1024> ref_rms{}, got_rms{};
                for (int i = 0; i < N; ++i) ref_rms[i % 1024] = std::fma(x[i], x[i], ref_rms[i % 1024]);
                for (int tid = 0; tid < 1024; ++tid)
                    for (int i = tid; i < N; i += 1024) {
                        cached[i] = x[i];
                        got_rms[tid] = std::fma(cached[i], cached[i], got_rms[tid]);
                    }
                float rs = reduce<1024>(ref_rms), gs = reduce<1024>(got_rms);
                assert(bits(rs) == bits(gs));
                float inv_rms = 1.0f / std::sqrt(rs / float(N) + 1e-20f);
                std::array<float, 24> ref_mix{}, got_mix{};
                for (int row = 0; row < R; ++row) {
                    ref_mix[row] = reference_dot(x, fn.data() + row * N) * inv_rms;
                    got_mix[row] = candidate_dot(cached, fn.data() + row * N) * inv_rms;
                    assert(bits(ref_mix[row]) == bits(got_mix[row]));
                }
                auto a = finish(ref_mix), b = finish(got_mix);
                for (int i = 0; i < 24; ++i) { assert(bits(a[i]) == bits(b[i])); assert(std::isfinite(a[i])); }
                float pin[4] = {normal(rng), normal(rng), normal(rng), normal(rng)};
                std::vector<float> ref_y(D), got_y(D);
                for (int d = 0; d < D; ++d) {
                    float s = 0;
                    for (int j = 0; j < 4; ++j) s = std::fma(pin[j], x[j * D + d], s);
                    ref_y[d] = bf16(s);
                }
                for (int tid = 768; tid < 1024; ++tid)
                    for (int d = tid - 768; d < D; d += 256) {
                        float s = 0;
                        for (int j = 0; j < 4; ++j) s = std::fma(pin[j], cached[j * D + d], s);
                        got_y[d] = bf16(s);
                    }
                for (int d = 0; d < D; ++d) assert(bits(ref_y[d]) == bits(got_y[d]));
                ++tokens;
            }
        }
    }
    std::printf("PASS CPU model: %d tokens, m=1..8, seven finite-data patterns; bitwise RMS/dots/coefficients/y; exact ownership and fixed-dimension bounds.\n", tokens);
    std::puts("GPU correctness, CUDA rsqrt/exp behavior, stream/capture execution, races, and timing remain untested.");
}
