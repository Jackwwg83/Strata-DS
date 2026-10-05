// CPU semantic checks only. This does not execute or time the CUDA kernels.
// g++ -std=c++17 -O3 -march=native host_semantics.cpp -o /tmp/k8_semantics
#include "math.hpp"
#include "layout.hpp"
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
constexpr int N = 384, D = 5120, K = 6, T = 128;
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
    constexpr int E = kd::kExpertsPerBlock, W = kd::kWeightTile;
    require(m >= 1 && m <= 8 && D == kd::kDimension, "fixed interface mismatch");
    require(kd::shared_bytes(m) <= 99 * 1024, "shared memory limit exceeded");
    const int threads = E * m * 32;
    require(threads <= 1024, "thread limit exceeded");
    std::vector<float> logits(m * N);
    std::vector<int> outputs(m * N, 0), weight_reads(N * D, 0);
    for (int block = 0; block < N / E; ++block) {
        std::vector<float> cached_x(m * D), tile_w(E * W);
        std::vector<int> x_writes(m * D, 0), tile_epoch(E * W, -1), lane_next(E * m * 32);
        std::vector<float> accum(threads, 0);
        for (int thread = 0; thread < threads; ++thread) {
            lane_next[thread] = thread & 31;
            for (int i = thread; i < m * D; i += threads) {
                require(++x_writes[i] == 1, "multiple activation producers");
                cached_x[i] = x[i];
            }
        }
        for (int count : x_writes) require(count == 1, "missing activation producer");
        int retired = -1;
        for (int base = 0, epoch = 0; base < D; base += W, ++epoch) {
            const int count = std::min(W, D - base);
            require(retired == epoch - 1, "weight overwrite before consumer retirement");
            std::vector<int> phase_writes(E * W, 0);
            // Mirror the actual producer assignment, including the short tail.
            for (int thread = 0; thread < threads; ++thread) {
                for (int i = thread; i < E * W; i += threads) {
                    const int row = i / W, d = i % W, slot = i;
                    if (d >= count) continue;
                    const int wi = (block * E + row) * D + base + d;
                    require(++phase_writes[slot] == 1, "multiple weight producers");
                    require(++weight_reads[wi] == 1, "weight reread across tokens");
                    tile_w[slot] = w[wi];
                    tile_epoch[slot] = epoch;
                }
            }
            // CTA publication barrier: all producers precede every consumer.
            for (int thread = 0; thread < threads; ++thread) {
                const int lane = thread & 31, warp = thread >> 5;
                const int token = warp % m, row = warp / m;
                for (int d = lane; d < count; d += 32) {
                    const int slot = row * W + d;
                    require(tile_epoch[slot] == epoch && phase_writes[slot] == 1,
                            "read of unpublished or padded weight");
                    require(base + d == lane_next[thread], "FMA lane order changed");
                    lane_next[thread] += 32;
                    accum[thread] = std::fma(cached_x[token * D + base + d], tile_w[slot], accum[thread]);
                }
            }
            // CTA retirement barrier: all consumers precede the next producer.
            retired = epoch;
        }
        for (int thread = 0; thread < threads; ++thread)
            require(lane_next[thread] == D + (thread & 31), "missing lane contribution");
        for (int offset = 16; offset > 0; offset >>= 1) {
            const auto old = accum;
            for (int thread = 0; thread < threads; ++thread)
                if ((thread & 31) + offset < 32) accum[thread] = old[thread] + old[thread + offset];
        }
        for (int warp = 0; warp < E * m; ++warp) {
            const int token = warp % m, expert = block * E + warp / m;
            const int oi = token * N + expert;
            require(++outputs[oi] == 1, "multiple score owners");
            logits[oi] = accum[warp * 32];
        }
    }
    for (int count : outputs) require(count == 1, "missing score owner");
    for (int count : weight_reads) require(count == 1, "weight missing or duplicated");
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
        std::array<Candidate, 32> warp_winners;
        warp_winners.fill({-INFINITY, N});
        for (int warp = 0; warp < T / 32; ++warp) {
            std::array<Candidate, 32> lanes;
            std::copy_n(local.begin() + warp * 32, 32, lanes.begin());
            warp_winners[warp] = warp_reduce(lanes);
        }
        const int winner = warp_reduce(warp_winners).id;
        require(winner < N, "selection produced sentinel");
        result.ids[i] = winner;
        selected[i] = raw[winner];
        values[winner] = {-INFINITY, N};
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
        require(error <= 1e-5 * std::abs(double(ref.weights[i])), "weight tolerance exceeded");
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
    // Cancellation crosses both streamed weight boundaries. These values are
    // exactly BF16-representable; a separate partial sum per phase is invalid.
    for (int m : {1, 3, 8}) {
        std::vector<float> x(m * D, 1.0f), w(N * D, 0.0f);
        for (int e = 0; e < N; ++e) {
            w[e * D] = std::ldexp(1.0f, 24);
            for (int d = 32; d < 2048; d += 32) w[e * D + d] = 1.0f;
            w[e * D + 2048] = -std::ldexp(1.0f, 24);
            w[e * D + 4096] = float((e % 7) - 3);
            w[e * D + 5056] = 0.125f;
        }
        const auto tiled = tile_logits(x, w, m);
        for (int t = 0; t < m; ++t) {
            std::vector<float> reference(N);
            for (int e = 0; e < N; ++e) {
                reference[e] = reference_dot(x.data() + t * D, w.data() + e * D);
                require(std::memcmp(&reference[e], &tiled[t * N + e], sizeof(float)) == 0,
                        "cancellation changed across phase boundary");
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
    // Scores that underflow in float but remain distinct in double must win
    // against a small positive bias on an even smaller score.
    logits.assign(N, -200.0f); b.assign(N, 1e-24f);
    for (int e = 0; e < K; ++e) { logits[e] = -104.0f; b[e] = 0.0f; }
    const auto underflow = compare(logits, b);
    for (int i = 0; i < K; ++i) require(underflow.ids[i] == i, "underflow order changed");
    // All negative-infinity comparison keys still select six real lower IDs.
    logits.assign(N, 0.0f); b.assign(N, -INFINITY);
    const auto minus_inf = compare(logits, b);
    for (int i = 0; i < K; ++i) require(minus_inf.ids[i] == i, "sentinel displaced real expert");
    // Mixed underflow, large positive scores, negative bias and repeated ties.
    for (unsigned seed = 100; seed < 1100; ++seed) {
        logits = random_values(N, seed % 2 ? 120.0f : 3.0f, seed, false);
        b = random_values(N, 2.0f, seed + 1000, false);
        for (int e = 0; e < N; e += 17) { logits[e] = 0; b[e] = 0.5f; }
        compare(logits, b);
    }
    std::printf("PASS CPU semantics: %d routing cases, %d bit-exact FP32 logits, max relative weight error %.3g\n", cases, logits_checked, max_relative_error);
    std::puts("Covers all m=1..8, once-per-CTA activation ownership, once-per-call weight reads, score ownership, three phase publication/retirement barriers and short tail.");
    std::puts("Bit-exact random/cancellation lane chains; exact/near ties, negative-infinity bias, softplus edges and float-underflow ordering; <=96 KiB shared memory.");
    std::puts("GPU execution, CUDA libm parity, graph capture/replay, and timings remain untested.");
}

#endif
