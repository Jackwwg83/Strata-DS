// CPU semantic checks only. This does not execute or time the CUDA kernels.
// g++ -std=c++17 -O3 -march=native host_semantics.cpp -o /tmp/k8_semantics
#include "math.hpp"
#include "pipeline.hpp"
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
    constexpr int Width = kd::kTile, Experts = kd::kExpertsPerBlock;
    constexpr int ChunksPerRow = Width / kd::kCopyValues;
    const int rows = Experts + m, threads = Experts * m * 32;
    const int values = rows * Width;
    long weight_reads = 0, input_reads = 0;
    std::vector<float> logits(m * N);
    for (int block = 0; block < N / Experts; ++block) {
        std::array<std::vector<float>, 2> shared;
        for (auto& buffer : shared) buffer.resize(values, NAN);
        std::vector<float> pending(values);
        std::array<int, 2> published_base{-1, -1};
        std::array<bool, 2> readers_finished{true, true};
        int pending_buffer = -1, pending_base = -1;
        auto stage = [&](int buffer, int base) {
            require(pending_buffer == -1, "overwriting an in-flight copy group");
            require(readers_finished[buffer], "producer overwrites an active consumer buffer");
            std::vector<int> owners(rows * ChunksPerRow, 0);
            for (int thread = 0; thread < threads; ++thread) {
                for (int chunk = thread; chunk < rows * ChunksPerRow; chunk += threads) {
                    ++owners[chunk];
                    const int row = chunk / ChunksPerRow;
                    const int d = (chunk % ChunksPerRow) * kd::kCopyValues;
                    for (int i = 0; i < kd::kCopyValues; ++i) {
                        const int source = (row < Experts ? block * Experts + row : row - Experts) * D + base + d + i;
                        pending[row * Width + d + i] = row < Experts ? w.at(source) : x.at(source);
                        if (row < Experts) ++weight_reads; else ++input_reads;
                    }
                }
            }
            require(std::all_of(owners.begin(), owners.end(), [](int n) { return n == 1; }), "copy ownership is not one-to-one");
            pending_buffer = buffer;
            pending_base = base;
        };
        auto wait_and_publish = [&] {
            if (pending_buffer >= 0) {
                shared[pending_buffer] = pending;
                published_base[pending_buffer] = pending_base;
                readers_finished[pending_buffer] = false;
                pending_buffer = -1;
            }
        };
        std::array<std::array<std::array<float, 32>, 8>, Experts> accum{};
        stage(0, 0);
        wait_and_publish();
        for (int base = 0, current = 0; base < D; base += Width, current ^= 1) {
            if (base + Width < D) stage(current ^ 1, base + Width);
            require(published_base[current] == base, "consumer reads unpublished or stale tile");
            for (int d = 0; d < Width; d += 32)
                for (int e = 0; e < Experts; ++e)
                    for (int t = 0; t < m; ++t)
                        for (int lane = 0; lane < 32; ++lane)
                            accum[e][t][lane] = std::fma(shared[current][(Experts + t) * Width + d + lane],
                                                       shared[current][e * Width + d + lane], accum[e][t][lane]);
            readers_finished[current] = true;
            wait_and_publish();
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const auto old = accum;
            for (int e = 0; e < Experts; ++e)
                for (int t = 0; t < m; ++t)
                    for (int lane = 0; lane + offset < 32; ++lane)
                        accum[e][t][lane] = old[e][t][lane] + old[e][t][lane + offset];
        }
        for (int e = 0; e < Experts; ++e)
            for (int t = 0; t < m; ++t) logits[t * N + block * Experts + e] = accum[e][t][0];
    }
    require(weight_reads == long(N) * D, "weight was not read once across all tokens");
    require(input_reads == long(N / Experts) * m * D, "input tile reuse differs");
    return logits;
}
void check_copy_alignment() {
    // Every legal BF16 base alignment, each tile, and every 16-byte copy. All
    // row/tile strides are multiples of 16, so the condition is row-invariant.
    for (int offset = 0; offset < 16; offset += 2)
        for (int row = 0; row < N; ++row)
            for (int base = 0; base < D; base += kd::kTile)
                for (int d = 0; d < kd::kTile; d += kd::kCopyValues) {
                    const uintptr_t src = offset + 2 * (row * D + base + d);
                    const uintptr_t dst = 2 * d;
                    require((dst & 15) == 0, "shared chunk is not 16-byte aligned");
                    require(((src & 15) == 0) == (offset == 0), "copy selected a misaligned async transaction");
                    require(d + kd::kCopyValues <= kd::kTile, "chunk crosses tile boundary");
                }
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
    check_copy_alignment();
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
    logits.assign(N, 1);
    b.assign(N, -INFINITY);
    auto negative_infinity = compare(logits, b);
    for (int i = 0; i < K; ++i) require(negative_infinity.ids[i] == i, "sentinel repeated an expert with -infinity bias");
    // Mixed underflow, large positive scores, negative bias and repeated ties.
    for (unsigned seed = 100; seed < 1100; ++seed) {
        logits = random_values(N, seed % 2 ? 120.0f : 3.0f, seed, false);
        b = random_values(N, 2.0f, seed + 1000, false);
        for (int e = 0; e < N; e += 17) { logits[e] = 0; b[e] = 0.5f; }
        compare(logits, b);
    }
    std::printf("PASS CPU semantics: %d routing cases, %d bit-exact FP32 logits, max relative weight error %.3g\n", cases, logits_checked, max_relative_error);
    std::puts("Covers m=1..8, ping-pong publication/reuse, unique chunk ownership, one weight read across tokens, all BF16 base alignments, ties, near-ties, threshold neighbors, large and underflow logits.");
    std::puts("GPU execution, CUDA libm parity, graph capture/replay, and timings remain untested.");
}

#endif
