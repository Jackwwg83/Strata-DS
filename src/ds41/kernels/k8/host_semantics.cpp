// CPU semantic checks only. This does not execute or time the CUDA kernels.
// g++ -std=c++17 -O3 -march=native host_semantics.cpp -o /tmp/k8_semantics
#include "math.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace kd = strata::ds41::kernels::k8_detail;
constexpr int N = 384, D = 5120, K = 6, T = 32;
struct Result { std::array<int, K> ids; std::array<float, K> weights; };
struct Candidate { double value; int id; };
static int cases = 0, logits_checked = 0;
static double max_relative_error = 0;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
float bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
std::vector<float> random_values(int n, float scale, unsigned seed, bool quantize) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0f, scale);
    std::vector<float> v(n);
    for (float& x : v) { x = distribution(generator); if (quantize) x = bf16(x); }
    return v;
}
float reference_dot(const float* x, const float* w) {
    std::array<float, 32> accum{};
    for (int lane = 0; lane < 32; ++lane)
        for (int d = lane; d < D; d += 32) accum[lane] = std::fma(x[d], w[d], accum[lane]);
    for (int offset = 16; offset > 0; offset >>= 1) {
        const auto old = accum;
        for (int lane = 0; lane + offset < 32; ++lane) accum[lane] = old[lane] + old[lane + offset];
    }
    return accum[0];
}
std::vector<float> tile_logits(const std::vector<float>& x, const std::vector<float>& w, int m) {
    std::vector<float> logits(m * N);
    for (int e = 0; e < N; ++e) {
        std::array<float, D> shared;
        // Mirror cooperative global-to-shared row staging, with unique owners.
        const int threads = m <= 4 ? 128 : 256;
        for (int thread = 0; thread < threads; ++thread)
            for (int d = thread; d < D; d += threads) shared[d] = w[e * D + d];
        std::array<std::array<float, 32>, 8> accum{};
        for (int base = 0; base < D; base += 32)
            for (int t = 0; t < m; ++t)
                for (int lane = 0; lane < 32; ++lane)
                    accum[t][lane] = std::fma(x[t * D + base + lane], shared[base + lane], accum[t][lane]);
        for (int offset = 16; offset > 0; offset >>= 1) {
            const auto old = accum;
            for (int t = 0; t < m; ++t)
                for (int lane = 0; lane + offset < 32; ++lane)
                    accum[t][lane] = old[t][lane] + old[t][lane + offset];
        }
        for (int t = 0; t < m; ++t) logits[t * N + e] = accum[t][0];
    }
    return logits;
}
Result oracle(const std::vector<float>& logits, const std::vector<float>& bias) {
    std::array<double, N> raw{}, values{};
    std::array<int, N> order{};
    for (int e = 0; e < N; ++e) {
        const double z = logits[e];
        raw[e] = std::sqrt(z > 20 ? z : std::log1p(std::exp(z)));
        values[e] = raw[e] + bias[e];
    }
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](int a, int b) {
        return values[a] > values[b] || (values[a] == values[b] && a < b);
    });
    Result result;
    double sum = 0;
    for (int i = 0; i < K; ++i) sum += raw[order[i]];
    for (int i = 0; i < K; ++i) {
        result.ids[i] = order[i];
        result.weights[i] = float(raw[order[i]] / (sum + 1e-20) * 1.5);
    }
    return result;
}
Candidate warp_reduce(std::array<Candidate, 32> values) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const auto old = values;
        for (int lane = 0; lane + offset < 32; ++lane) {
            const auto a = old[lane + offset], b = old[lane];
            if (kd::better(a.value, a.id, b.value, b.id)) values[lane] = a;
        }
    }
    return values[0];
}
Result simulated_select(const std::vector<float>& logits, const std::vector<float>& bias) {
    std::array<double, N> raw{};
    std::array<Candidate, N> values{};
    for (int e = 0; e < N; ++e) {
        raw[e] = kd::score(logits[e]);
        values[e] = {raw[e] + double(bias[e]), e};
    }
    Result result;
    std::array<double, K> selected{};
    for (int i = 0; i < K; ++i) {
        std::array<Candidate, T> local;
        for (int thread = 0; thread < T; ++thread) {
            Candidate best{-INFINITY, N};
            for (int j = 0; j < N / T; ++j) {
                const Candidate c = values[thread + j * T];
                if (kd::better(c.value, c.id, best.value, best.id)) best = c;
            }
            local[thread] = best;
        }
        const int winner = warp_reduce(local).id;
        require(winner < N, "selection produced sentinel");
        result.ids[i] = winner;
        std::array<double, T> owner_scores{};
        for (int lane = 0; lane < T; ++lane) {
            for (int j = 0; j < N / T; ++j) {
                const int slot = lane + j * T;
                if (values[slot].id == winner) {
                    owner_scores[lane] = raw[slot];
                    values[slot] = {-INFINITY, N};
                }
            }
        }
        // Mirror the source-lane shuffle and register in output lane i.
        selected[i] = owner_scores[winner & (T - 1)];
    }
    double sum = 0;
    for (double s : selected) sum += s;
    for (int i = 0; i < K; ++i) result.weights[i] = float(selected[i] / (sum + 1e-20) * 1.5);
    return result;
}
Result compare(const std::vector<float>& logits, const std::vector<float>& bias) {
    const auto ref = oracle(logits, bias), got = simulated_select(logits, bias);
    require(ref.ids == got.ids, "expert IDs/order differ");
    for (int i = 0; i < K; ++i) {
        require(std::isfinite(got.weights[i]), "nonfinite output weight");
        const double error = std::abs(double(got.weights[i]) - ref.weights[i]);
        require(error == 0, "CPU normalization bits differ");
        if (ref.weights[i] != 0) max_relative_error = std::max(max_relative_error, error / std::abs(double(ref.weights[i])));
    }
    ++cases;
    return got;
}
#ifndef K8_SEMANTICS_LIBRARY
int main() {
    const auto w = random_values(N * D, 0.02f, 1, true);
    const auto bias = random_values(N, 0.1f, 2, false);
    for (int m = 1; m <= 8; ++m) {
        const auto x = random_values(m * D, 1.0f, 10 + m, true);
        const auto tiled = tile_logits(x, w, m);
        for (int t = 0; t < m; ++t) {
            std::vector<float> reference(N);
            for (int e = 0; e < N; ++e) {
                reference[e] = reference_dot(x.data() + t * D, w.data() + e * D);
                require(std::memcmp(&reference[e], &tiled[t * N + e], sizeof(float)) == 0, "FP32 logit bits differ");
                ++logits_checked;
            }
            compare(reference, bias);
        }
    }
    std::vector<float> logits(N, 0), b(N, 0);
    auto equal = compare(logits, b);
    for (int i = 0; i < K; ++i) require(equal.ids[i] == i, "all-equal tie did not select lower IDs");
    // A bias step erased by FP32 score+bias, but retained by the fixed oracle.
    b[383] = std::ldexp(1.0f, -26);
    require(float(kd::score(0) + b[383]) == float(kd::score(0)), "near-tie fixture not rounded away in FP32");
    require(compare(logits, b).ids[0] == 383, "near-tie incorrectly collapsed");
    b.assign(N, 0);
    std::fill(logits.begin(), logits.end(), -1000);
    auto zero = compare(logits, b);
    for (int i = 0; i < K; ++i) require(zero.ids[i] == i && zero.weights[i] == 0, "all-zero score behavior differs");
    const std::array<float, 16> edges = {
        -1000, -746, -745, -700, -100, -20, -1, -0.0f, 0, 1,
        std::nextafter(20.0f, 0.0f), 20, std::nextafter(20.0f, INFINITY),
        100, 1e20f, std::numeric_limits<float>::max()};
    for (float z : edges) { logits.assign(N, z); compare(logits, b); }
    for (int e = 0; e < N; ++e) { logits[e] = edges[e % edges.size()]; b[e] = float((e % 11) - 5) * 0.125f; }
    compare(logits, b);
    // Mixed underflow, large positive scores, negative bias and repeated ties.
    for (unsigned seed = 100; seed < 1100; ++seed) {
        logits = random_values(N, seed % 2 ? 120.0f : 3.0f, seed, false);
        b = random_values(N, 2.0f, seed + 1000, false);
        for (int e = 0; e < N; e += 17) { logits[e] = 0; b[e] = 0.5f; }
        compare(logits, b);
    }
    // Every expert visits the top rank: checks all 32 owners and 12 registers.
    logits.assign(N, 0);
    b.assign(N, 0);
    for (int e = 0; e < N; ++e) {
        b[e] = std::ldexp(1.0f, -26);
        require(compare(logits, b).ids[0] == e, "winner owner lane/register differs");
        b[e] = 0;
    }
    // All six winners can belong to one lane; removal must update its state.
    for (int lane = 0; lane < T; ++lane) {
        logits.assign(N, -1000);
        b.assign(N, 0);
        for (int i = 0; i < K; ++i) logits[lane + i * T] = float(100 - i);
        const auto same_owner = compare(logits, b);
        for (int i = 0; i < K; ++i)
            require(same_owner.ids[i] == lane + i * T, "same-lane winner removal differs");
    }
    // Bias may dominate or saturate comparisons but never enters normalization.
    logits.assign(N, -1000);
    b.assign(N, -std::numeric_limits<float>::infinity());
    const auto negative_infinity = compare(logits, b);
    for (int i = 0; i < K; ++i)
        require(negative_infinity.ids[i] == i && negative_infinity.weights[i] == 0,
                "negative-infinity ties selected removed sentinel");
    const std::array<float, 5> bias_edges = {
        -std::numeric_limits<float>::max(), -1e20f, 0, 1e20f,
        std::numeric_limits<float>::max()};
    for (float bias_edge : bias_edges) {
        b.assign(N, bias_edge);
        for (int e = 0; e < N; ++e) logits[e] = edges[e % edges.size()];
        compare(logits, b);
    }
    // Carry each original expert's score and bias through random permutations.
    // Unique scores must preserve rank/weights after mapping IDs back; ties are
    // independently checked against the new physical expert IDs by the oracle.
    std::mt19937 shuffle_rng(80207);
    std::array<int, N> perm;
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<float> original_logits(N), original_bias(N), pl(N), pb(N);
    for (int e = 0; e < N; ++e) original_logits[e] = float(e - N / 2) * 0.125f;
    const auto unique = compare(original_logits, original_bias);
    for (int trial = 0; trial < 512; ++trial) {
        std::shuffle(perm.begin(), perm.end(), shuffle_rng);
        for (int e = 0; e < N; ++e) {
            pl[e] = original_logits[perm[e]];
            pb[e] = original_bias[perm[e]];
        }
        const auto shuffled = compare(pl, pb);
        for (int i = 0; i < K; ++i) {
            require(perm[shuffled.ids[i]] == unique.ids[i], "permutation changed unique ranking");
            require(shuffled.weights[i] == unique.weights[i], "permutation changed unique weights");
        }
        for (int e = 0; e < N; ++e) {
            pl[e] = edges[perm[e] % edges.size()];
            pb[e] = float((perm[e] % 7) - 3) * 0.125f;
        }
        compare(pl, pb);
    }
    std::printf("PASS CPU semantics: %d routing cases, %d bit-exact FP32 logits, max relative weight error %.3g\n", cases, logits_checked, max_relative_error);
    std::puts("Covers all m=1..8, exact ties, FP32-collapsed near-ties, softplus threshold neighbors, large positive and underflow logits, all owner lanes/registers, bias extremes and 1,024 permutation cases.");
    std::puts("GPU execution, CUDA libm parity, graph capture/replay, and timings remain untested.");
}

#endif
