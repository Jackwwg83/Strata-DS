// CPU-only tests of the exact helper used by the CUDA scorer.
// g++ -std=c++17 -O3 -march=native check_register_prefetch.cpp -o /tmp/k8_prefetch
#define K8_SEMANTICS_LIBRARY 1
#include "host_semantics.cpp"
#include "register_prefetch.hpp"

struct CpuFma {
    float operator()(float a, float b, float c) const { return std::fma(a, b, c); }
};
struct CpuPair {
    const float* x;
    const float* w;
    int lane;
    kd::Pair operator()(int step) const {
        require(step >= 0 && step < D / 32, "out-of-bounds prefetch");
        const int d = lane + step * 32;
        return {x[d], w[d]};
    }
};
float prefetched_dot(const float* x, const float* w) {
    std::array<float, 32> lanes{};
    for (int lane = 0; lane < 32; ++lane)
        lanes[lane] = kd::register_prefetch<D / 32>(CpuPair{x, w, lane}, CpuFma{});
    for (int offset = 16; offset > 0; offset >>= 1) {
        const auto old = lanes;
        for (int lane = 0; lane + offset < 32; ++lane)
            lanes[lane] = old[lane] + old[lane + offset];
    }
    return lanes[0];
}
void same(float a, float b) {
    require(std::memcmp(&a, &b, sizeof(float)) == 0, "prefetch changed FP32 bits");
}
void check_schedule() {
    // Values encode their step, so any repeated, missing or reordered consume
    // fails independently of floating-point behavior. Observe loads and FMAs.
    for (int lane = 0; lane < 32; ++lane) {
        std::vector<int> loads, consumes;
        const float result = kd::register_prefetch<160>(
            [&](int step) {
                require(step >= 0 && step < 160, "pipeline read beyond row");
                require(step == int(loads.size()), "load order/duplication");
                loads.push_back(step);
                return kd::Pair{float(lane + step * 32), float(step)};
            },
            [&](float x, float w, float sum) {
                const int step = int(consumes.size());
                require(x == lane + step * 32 && w == step, "consume order/duplication");
                require(int(loads.size()) == std::min(step + 2, 160), "load-ahead/drain schedule");
                consumes.push_back(step);
                return sum + 1.0f;
            });
        require(loads.size() == 160 && consumes.size() == 160 && result == 160,
                "missing lane-stride step or drain");
    }
    std::puts("PASS schedule: all 32 lanes load and consume exactly 160 steps, one-step load-ahead and final drain");
}
int main() {
    check_schedule();
    int checked = 0;
    const auto w = random_values(N * D, 0.02f, 19001, true);
    const auto bias = random_values(N, 0.1f, 19002, false);
    for (int m = 1; m <= 8; ++m) {
        const auto x = random_values(m * D, 1.0f, 19100 + m, true);
        std::vector<std::vector<float>> logits(m, std::vector<float>(N));
        for (int e = 0; e < N; ++e) {
            std::array<float, D> shared{};
            // Same single-owner cooperative staging as the actual m=2..8 path.
            const int threads = m <= 4 ? 128 : 256;
            for (int thread = 0; thread < threads; ++thread)
                for (int d = thread; d < D; d += threads) shared[d] = w[e * D + d];
            for (int t = 0; t < m; ++t) {
                const float* xp = x.data() + t * D;
                const float* wp = w.data() + e * D;
                const float expected = reference_dot(xp, wp);
                same(expected, prefetched_dot(xp, wp));
                same(expected, prefetched_dot(xp, shared.data()));
                logits[t][e] = expected;
                checked += 2;
            }
        }
        for (int t = 0; t < m; ++t) compare(logits[t], bias);
    }
    // Exact BF16 factors trigger catastrophic cancellation if terms are split
    // into multiple accumulators or consumed in a different order.
    std::vector<float> x(D, 0), row(D, 0);
    int cancellation = 0;
    for (int lane = 0; lane < 32; ++lane) {
        for (int start : {0, 1, 6, 7, 8, 77, 78, 79, 80, 151, 152, 156, 157}) {
            for (int sign : {-1, 1}) {
                x.assign(D, 0); row.assign(D, 0);
                const int d = lane + start * 32;
                x[d] = float(sign) * 0x1p60f; row[d] = 0x1p60f;
                x[d + 32] = 1; row[d + 32] = 1;
                x[d + 64] = -float(sign) * 0x1p60f; row[d + 64] = 0x1p60f;
                const float reference = reference_dot(x.data(), row.data());
                require(reference == 0.0f, "cancellation fixture does not lose the middle term");
                same(reference, prefetched_dot(x.data(), row.data()));
                ++cancellation;
            }
        }
        // Nonzero last-step drain: two huge opposite products, then one.
        x.assign(D, 0); row.assign(D, 0);
        x[lane + 157 * 32] = 0x1p60f; row[lane + 157 * 32] = 0x1p60f;
        x[lane + 158 * 32] = -0x1p60f; row[lane + 158 * 32] = 0x1p60f;
        x[lane + 159 * 32] = 1; row[lane + 159 * 32] = 1;
        same(1.0f, reference_dot(x.data(), row.data()));
        same(1.0f, prefetched_dot(x.data(), row.data()));
        ++cancellation;
    }
    int adversarial = 0;
    std::mt19937 rng(811);
    for (int trial = 0; trial < 1024; ++trial) {
        for (int d = 0; d < D; ++d) {
            x[d] = std::ldexp(float((int(rng() % 255) - 127)), int(rng() % 81) - 47);
            row[d] = std::ldexp(float((int(rng() % 255) - 127)), int(rng() % 81) - 47);
        }
        same(reference_dot(x.data(), row.data()), prefetched_dot(x.data(), row.data()));
        ++adversarial;
    }
    std::printf("PASS exact model: %d direct/shared FP32 logits, m=1..8, %d cancellation/drain fixtures, %d exponent/sign stress dots, %d routing comparisons\n", checked, cancellation, adversarial, cases);
    std::puts("CPU models only; GPU numerical, graph and performance checks remain pending.");
}
