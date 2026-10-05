// Standalone CPU model of K8-04. No CUDA execution or timing claims.
// g++ -std=c++17 -O3 -ffp-contract=off host_split_model.cpp -o /tmp/k8_split_model
#include "split_math.hpp"
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

namespace math = strata::ds41::kernels::k8_split;
constexpr int D = 5120, E = 384, TOP = 6;
struct Routing { std::array<int, TOP> ids; std::array<float, TOP> weights; };
struct Candidate { double value; int id; };
static int dot_checks = 0, route_checks = 0;
static double max_weight_error = 0;

void require(bool condition, const char* text) {
    if (!condition) throw std::runtime_error(text);
}
uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
float bf16(float value) {
    uint32_t b = bits(value);
    b = (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&value, &b, sizeof(value));
    return value;
}
float reference_dot(const float* x, const float* w) {
    std::array<float, 32> lanes{};
    for (int lane = 0; lane < 32; ++lane) {
        for (int d = lane; d < D; d += 32) lanes[lane] = std::fma(x[d], w[d], lanes[lane]);
    }
    for (int offset = 16; offset; offset /= 2) {
        const auto old = lanes;
        for (int lane = 0; lane + offset < 32; ++lane) lanes[lane] = old[lane] + old[lane + offset];
    }
    return lanes[0];
}
std::vector<float> split_logits(const std::vector<float>& x, const std::vector<float>& w, int m) {
    // Model the exact launch geometry, parity partition, token register tile,
    // half-warp reductions, scratch layout, and final addition in the kernel.
    std::vector<float> scratch(m * 2 * E, std::numeric_limits<float>::quiet_NaN());
    for (int part = 0; part < 2; ++part) {
        for (int block = 0; block < E / 2; ++block) {
            std::array<std::array<float, 32>, 8> sums{};
            for (int step = 0; step < D / 32; ++step) {
                for (int thread = 0; thread < 32; ++thread) {
                    const int expert = 2 * block + thread / 16;
                    const int d = 32 * step + 2 * (thread % 16) + part;
                    const float weight = w[expert * D + d];
                    for (int t = 0; t < m; ++t) sums[t][thread] = std::fma(x[t * D + d], weight, sums[t][thread]);
                }
            }
            for (int offset = 8; offset; offset /= 2) {
                const auto old = sums;
                for (int t = 0; t < m; ++t) {
                    for (int thread = 0; thread < 32; ++thread) {
                        const int source = thread % 16 + offset < 16 ? thread + offset : thread;
                        sums[t][thread] = old[t][thread] + old[t][source];
                    }
                }
            }
            for (int t = 0; t < m; ++t) {
                for (int half = 0; half < 2; ++half) scratch[(t * 2 + part) * E + 2 * block + half] = sums[t][16 * half];
            }
        }
    }
    std::vector<float> logits(m * E);
    for (int t = 0; t < m; ++t) {
        for (int e = 0; e < E; ++e) logits[t * E + e] = scratch[(t * 2) * E + e] + scratch[(t * 2 + 1) * E + e];
    }
    return logits;
}
float contiguous_split_dot(const float* x, const float* w) {
    // Deliberately unsafe control: restart every lane's FMA chain at K/2.
    float partials[2] = {};
    for (int part = 0; part < 2; ++part) {
        std::array<float, 32> lanes{};
        for (int lane = 0; lane < 32; ++lane) {
            for (int d = part * (D / 2) + lane; d < (part + 1) * (D / 2); d += 32) lanes[lane] = std::fma(x[d], w[d], lanes[lane]);
        }
        for (int offset = 16; offset; offset /= 2) {
            const auto old = lanes;
            for (int lane = 0; lane + offset < 32; ++lane) lanes[lane] = old[lane] + old[lane + offset];
        }
        partials[part] = lanes[0];
    }
    return partials[0] + partials[1];
}
Routing oracle(const std::vector<float>& logits, const std::vector<float>& bias) {
    std::array<double, E> scores{}, ranked{};
    std::array<int, E> order{};
    for (int e = 0; e < E; ++e) {
        const double z = logits[e];
        scores[e] = std::sqrt(z > 20 ? z : std::log1p(std::exp(z)));
        ranked[e] = scores[e] + double(bias[e]);
    }
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + TOP, order.end(), [&](int a, int b) {
        return ranked[a] > ranked[b] || (ranked[a] == ranked[b] && a < b);
    });
    Routing result{};
    double sum = 0;
    for (int i = 0; i < TOP; ++i) sum += scores[order[i]];
    for (int i = 0; i < TOP; ++i) {
        result.ids[i] = order[i];
        result.weights[i] = float(scores[order[i]] / (sum + 1e-20) * 1.5);
    }
    return result;
}
Candidate reduce(std::array<Candidate, 32> lanes) {
    for (int offset = 16; offset; offset /= 2) {
        const auto old = lanes;
        for (int lane = 0; lane + offset < 32; ++lane) {
            const auto other = old[lane + offset];
            if (math::better(other.value, other.id, old[lane].value, old[lane].id)) lanes[lane] = other;
        }
    }
    return lanes[0];
}
Routing modeled_select(const std::vector<float>& logits, const std::vector<float>& bias) {
    std::array<Candidate, E> candidates{};
    std::array<double, E> raw{};
    for (int e = 0; e < E; ++e) {
        raw[e] = math::score(logits[e]);
        candidates[e] = {raw[e] + double(bias[e]), e};
    }
    Routing result{};
    double selected[TOP] = {};
    for (int rank = 0; rank < TOP; ++rank) {
        std::array<Candidate, 32> winners;
        winners.fill({-INFINITY, E});
        for (int warp = 0; warp < E / 32; ++warp) {
            std::array<Candidate, 32> lanes;
            std::copy_n(candidates.begin() + warp * 32, 32, lanes.begin());
            winners[warp] = reduce(lanes);
        }
        const int winner = reduce(winners).id;
        require(winner < E, "winner was sentinel");
        result.ids[rank] = winner;
        selected[rank] = raw[winner];
        candidates[winner] = {-INFINITY, E};
    }
    double sum = 0;
    for (double score : selected) sum += score;
    for (int rank = 0; rank < TOP; ++rank) result.weights[rank] = float(selected[rank] / (sum + 1e-20) * 1.5);
    return result;
}
Routing check_route(const std::vector<float>& logits, const std::vector<float>& bias) {
    const auto reference = oracle(logits, bias), candidate = modeled_select(logits, bias);
    require(reference.ids == candidate.ids, "expert IDs/order differ");
    for (int i = 0; i < TOP; ++i) {
        const double error = std::abs(double(reference.weights[i]) - candidate.weights[i]);
        require(std::isfinite(candidate.weights[i]), "weight is nonfinite");
        require(error <= 1e-5 * std::abs(double(reference.weights[i])), "weight precision failed");
        if (reference.weights[i]) max_weight_error = std::max(max_weight_error, error / std::abs(double(reference.weights[i])));
    }
    ++route_checks;
    return candidate;
}
void check_tensor(const std::vector<float>& x, const std::vector<float>& w, int m, const std::vector<float>& bias) {
    const auto split = split_logits(x, w, m);
    for (int t = 0; t < m; ++t) {
        std::vector<float> logits(E);
        for (int e = 0; e < E; ++e) {
            logits[e] = reference_dot(x.data() + t * D, w.data() + e * D);
            require(bits(logits[e]) == bits(split[t * E + e]), "split changed FP32 dot bits");
            ++dot_checks;
        }
        check_route(logits, bias);
    }
}
std::vector<float> random_values(int count, unsigned seed, float scale, bool quantized) {
    std::mt19937 random(seed);
    std::normal_distribution<float> distribution(0, scale);
    std::vector<float> values(count);
    for (auto& value : values) { value = distribution(random); if (quantized) value = bf16(value); }
    return values;
}
#ifndef K8_SPLIT_MODEL_LIBRARY
int main() {
    const auto w = random_values(E * D, 1, 0.02f, true);
    const auto bias = random_values(E, 2, 0.1f, false);
    for (int m = 1; m <= 8; ++m) check_tensor(random_values(m * D, 10 + m, 1.0f, true), w, m, bias);

    // Cancellation adversary: the contiguous split produces 0, the reference
    // and exact-tree split produce 1. The changed value also changes top-six.
    std::vector<float> x(D, 1), adversarial_w(E * D, 0), zero_bias(E, 0);
    float* row = adversarial_w.data() + (E - 1) * D;
    row[0] = std::ldexp(1.0f, 25); row[32] = 1;
    row[D / 2] = -std::ldexp(1.0f, 25); row[D / 2 + 32] = 1;
    require(reference_dot(x.data(), row) == 1 && contiguous_split_dot(x.data(), row) == 0,
            "cancellation fixture did not expose unsafe reassociation");
    check_tensor(x, adversarial_w, 1, zero_bias);
    std::vector<float> correct_logits(E, 0), unsafe_logits(E, 0);
    correct_logits[E - 1] = 1;
    require(oracle(correct_logits, zero_bias).ids != oracle(unsafe_logits, zero_bias).ids,
            "unsafe split control failed to change top-six IDs");

    // Stress every original lane and both partitions, at both K halves. The
    // even/odd groups receive differently signed cancellation and tiny terms.
    for (int e = 0; e < E; ++e) {
        float* r = adversarial_w.data() + e * D;
        std::fill_n(r, D, 0);
        for (int lane = 0; lane < 32; ++lane) {
            const float sign = ((e + lane) & 1) ? -1.0f : 1.0f;
            r[lane] = sign * std::ldexp(1.0f, 24 + (e % 8));
            r[32 + lane] = float((e + lane) % 3 - 1);
            r[D / 2 + lane] = -r[lane];
            r[D / 2 + 32 + lane] = sign * std::ldexp(1.0f, -10 + (e % 8));
        }
    }
    for (int m : {1, 2, 5, 8}) {
        std::vector<float> ones(m * D, 1);
        for (int t = 0; t < m; ++t) for (int d = 0; d < D; ++d) ones[t * D + d] = (t & 1) ? -1.0f : 1.0f;
        check_tensor(ones, adversarial_w, m, bias);
    }

    std::vector<float> logits(E, 0), b(E, 0);
    const auto tied = check_route(logits, b);
    for (int i = 0; i < TOP; ++i) require(tied.ids[i] == i, "lower-ID exact tie failed");
    b[E - 1] = std::ldexp(1.0f, -26);
    require(float(math::score(0) + b[E - 1]) == float(math::score(0)), "near-tie fixture was not sub-FP32");
    require(check_route(logits, b).ids[0] == E - 1, "sub-FP32 near-tie lost");
    // Near ties at the inclusion boundary, including nonadjacent warp owners.
    b.assign(E, 0);
    for (int e : {3, 42, 117, 208, 333}) b[e] = 1;
    b[382] = std::ldexp(1.0f, -26);
    require(check_route(logits, b).ids[5] == 382, "sixth/seventh near-tie lost");

    const std::array<float, 18> edges = {-1000, -746, -745, -700, -100, -20, -1, -0.0f, 0, 1,
        std::nextafter(20.0f, 0.0f), 20, std::nextafter(20.0f, INFINITY), 100, 1e10f, 1e20f,
        std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min()};
    b.assign(E, 0);
    for (float value : edges) { logits.assign(E, value); check_route(logits, b); }
    logits.assign(E, -1000);
    const auto zero = check_route(logits, b);
    for (int i = 0; i < TOP; ++i) require(zero.ids[i] == i && zero.weights[i] == 0, "zero-score normalization failed");
    for (unsigned seed = 20; seed < 2020; ++seed) {
        logits = random_values(E, seed, seed & 1 ? 120.0f : 3.0f, false);
        b = random_values(E, seed + 2020, 1.0f, false);
        for (int e = 0; e < E; e += 13) { logits[e] = edges[seed % edges.size()]; b[e] = 0; }
        check_route(logits, b);
    }
    std::printf("PASS CPU model: %d bit-exact logits, %d routing cases; max relative weight error %.3g\n",
                dot_checks, route_checks, max_weight_error);
    std::puts("All m=1..8; exact and near ties; cancellation control; threshold/underflow/extreme finite scores.");
    std::puts("CUDA execution/libm parity, graph capture/replay, and GPU timings remain untested.");
}
#endif  // K8_SPLIT_MODEL_LIBRARY
