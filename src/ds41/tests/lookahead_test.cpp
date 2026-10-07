// src/ds41/tests/lookahead_test.cpp - the router lookahead: cpu_router_topk picks the experts a double-precision
// reference picks (except where two scores tie within 1e-4 at the k-th place), and RouterLookahead warms the next
// layer's predicted experts and counts the ones the layer then used.
#include "strata/ds41/lookahead.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace strata::ds41;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what.c_str()); }
}

uint16_t to_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return (uint16_t) ((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
float from_bf16(uint16_t v) { uint32_t b = (uint32_t) v << 16; float f; std::memcpy(&f, &b, 4); return f; }
uint16_t to_f16(float f) {   // normal range only, round to nearest
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const uint32_t s = (b >> 16) & 0x8000u;
    const int e = (int) ((b >> 23) & 0xff) - 127 + 15;
    if (e <= 0) return (uint16_t) s;
    const uint32_t m = (b & 0x7fffff) + 0x1000;
    return (uint16_t) (s | ((uint32_t) (e + (m >> 23)) << 10) | ((m & 0x7fffff) >> 13));
}
float from_f16(uint16_t h) {
    const int e = (h >> 10) & 0x1f;
    const float m = (float) (h & 0x3ff);
    const float v = e == 0 ? m * std::ldexp(1.f, -24) : std::ldexp(1.f + m / 1024.f, e - 15);
    return h & 0x8000 ? -v : v;
}

}  // namespace

int main() {
    const int n = 384, dim = 5120, k = 6;
    std::mt19937 g(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> w((size_t) n * dim);
    for (auto& v : w) v = to_bf16(nd(g) * 0.02f);
    std::vector<float> bias(n);
    for (auto& v : bias) v = nd(g) * 0.05f;

    // 1. selection against a double-precision reference
    int exact = 0, near_tie = 0;
    const int trials = 50;
    for (int t = 0; t < trials; ++t) {
        std::vector<uint16_t> x(dim);
        for (auto& v : x) v = to_f16(nd(g));
        int32_t ids[k];
        cpu_router_topk(x.data(), w.data(), bias.data(), n, dim, k, ids);
        std::vector<double> s(n);
        for (int e = 0; e < n; ++e) {
            double acc = 0;
            for (int i = 0; i < dim; ++i) acc += (double) from_f16(x[i]) * from_bf16(w[(size_t) e * dim + i]);
            const double sp = acc > 20 ? acc : std::log1p(std::exp(acc));
            s[e] = std::sqrt(sp) + bias[e];
        }
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return s[a] > s[b]; });
        const std::set<int> want(order.begin(), order.begin() + k), got(ids, ids + k);
        if (want == got) ++exact;
        else if (std::fabs(s[order[k - 1]] - s[order[k]]) < 1e-4) ++near_tie;
        else check(false, "trial " + std::to_string(t) + ": a different top-k without a near tie");
    }
    std::printf("selection: %d/%d exact, %d near ties\n", exact, trials, near_tie);

    // 2. the lookahead thread: warms layer l+1's prediction; observe counts the used ones
    std::vector<std::vector<uint16_t>> ws(3, w);
    std::vector<std::vector<float>> bs(3, bias);
    std::mutex mu;
    std::vector<std::pair<int, int>> warmed;
    RouterLookahead la(ws, bs, n, dim, k, [&](int l, int e) {
        std::lock_guard<std::mutex> lk(mu);
        warmed.emplace_back(l, e);
        return e % 2 == 0;   // pretend only even experts live in the file tier
    });
    std::vector<uint16_t> x(dim);
    for (auto& v : x) v = to_f16(nd(g));
    int32_t pred[k];
    cpu_router_topk(x.data(), ws[1].data(), bs[1].data(), n, dim, k, pred);
    la.post(0, x.data());
    for (int i = 0; i < 200; ++i) {   // up to 2 s
        { std::lock_guard<std::mutex> lk(mu); if ((int) warmed.size() == k) break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    {
        std::lock_guard<std::mutex> lk(mu);
        check((int) warmed.size() == k, "the warm callback ran for each predicted expert");
        for (int i = 0; i < k && i < (int) warmed.size(); ++i)
            check(warmed[i] == std::make_pair(1, (int) pred[i]), "layer 1, the predicted ids in order");
    }
    int even = 0;
    for (int i = 0; i < k; ++i) even += pred[i] % 2 == 0;
    la.observe(1, pred, k);   // layer 1 used exactly the predicted experts
    const auto st = la.take_stats();
    check(st.predicted == k && st.warmed == even && st.useful == even, "stats: predicted, warmed, useful");
    la.post(2, x.data());   // the last layer: nothing to predict
    std::printf("RESULT %s\n", failures ? "fail" : "pass");
    return failures ? 1 : 0;
}
