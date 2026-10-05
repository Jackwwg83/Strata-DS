// src/ds41/tests/vram_experts_test.cu - the host logic of the VRAM expert tier: the STRP profile reader and the
// adaptive swap plan (upstream's rule). The slot copies and K10 are covered by the engine run.
#include "strata/ds41/vram_experts.hpp"

#include "bench_util.hpp"

#include <cstdio>
#include <functional>
#include <string>

using namespace ds41test;
using strata::ds41::ExpertSwap;
using strata::ds41::plan_expert_swaps;
using strata::ds41::read_expert_profile;

namespace {

void write_profile(const std::string& path, int nl, int ne, const std::vector<std::pair<int, int>>& pairs) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    const uint32_t n = (uint32_t) pairs.size();
    const uint32_t h[5] = {1, (uint32_t) nl, (uint32_t) ne, n, n};
    std::fwrite("STRP", 1, 4, f);
    std::fwrite(h, 4, 5, f);
    for (auto [l, e] : pairs) {
        const uint16_t p[2] = {(uint16_t) l, (uint16_t) e};
        std::fwrite(p, 2, 2, f);
    }
    std::vector<int32_t> table((size_t) nl * ne, -1);
    for (uint32_t i = 0; i < n; ++i) table[(size_t) pairs[i].first * ne + pairs[i].second] = (int32_t) i;
    std::fwrite(table.data(), 4, table.size(), f);
    std::fclose(f);
}

bool throws(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

}  // namespace

int main() {
    Verdict v;
    // ---- profile reader
    const std::string p = "/tmp/ds41_vram_experts_test.bin";
    write_profile(p, 3, 4, {{2, 1}, {0, 3}, {1, 0}});
    const auto r = read_expert_profile(p, 3, 4);
    v.check(r.size() == 3 && r[0] == std::make_pair(2, 1) && r[2] == std::make_pair(1, 0), "profile round trip");
    v.check(throws([&] { read_expert_profile(p, 3, 5); }), "profile of another shape is rejected");
    write_profile(p, 3, 4, {{2, 1}, {2, 1}});
    v.check(throws([&] { read_expert_profile(p, 3, 4); }), "a repeated pair is rejected");
    std::remove(p.c_str());

    // ---- swap plan: 2 layers x 6 experts
    const int L = 2, E = 6;
    std::vector<int32_t> res(L * E, -1);
    std::vector<float> u(L * E, 0.0f);
    // layer 0: resident e0 (usage 0) and e1 (5); missing e2 (3), e3 (1.9: below 2), e4 (2.0)
    res[0] = 0; res[1] = 1;
    u[0] = 0; u[1] = 5; u[2] = 3; u[3] = 1.9f; u[4] = 2.0f;
    auto s = plan_expert_swaps(u, res, L, E, 96);
    // e2 pairs with e0 (gain 3); e4 pairs with e1 next, 2.0 < 5 + 1.5: stop
    v.check(s.size() == 1 && s[0].layer == 0 && s[0].in == 2 && s[0].out == 0 && s[0].gain == 3.0f,
            "one swap: the most-used missing expert takes the least-used slot");
    // the margin: a candidate must lead by 1.5
    u[2] = 1.4f + 0.0f;
    u[0] = 0.6f;
    u[4] = 2.0f;   // 2.0 vs 0.6: lead 1.4 < 1.5
    s = plan_expert_swaps(u, res, L, E, 96);
    v.check(s.empty(), "no swap without a 1.5 lead");
    // layer 1: resident e0..e2 at usage 0; missing e3 (9), e4 (4), e5 (2); both layers, sorted by gain, capped
    res[6] = 2; res[7] = 3; res[8] = 4;
    u[9] = 9; u[10] = 4; u[11] = 2;
    u[2] = 7;   // layer 0: e2 (7) vs e0 (0.6): gain 6.4
    s = plan_expert_swaps(u, res, L, E, 96);
    v.check(s.size() == 4, "four swaps over two layers, got " + std::to_string(s.size()));
    if (s.size() == 4) {
        v.check(s[0].layer == 1 && s[0].in == 3 && s[0].gain == 9.0f, "largest gain first");
        v.check(s[1].layer == 0 && s[1].in == 2 && s[1].out == 0, "then layer 0");
        v.check(s[2].layer == 1 && s[2].in == 4 && s[3].in == 5, "then the rest of layer 1");
        v.check(s[2].out != s[3].out && s[0].out != s[2].out, "each victim used once");
    }
    s = plan_expert_swaps(u, res, L, E, 2);
    v.check(s.size() == 2 && s[0].gain == 9.0f && s[1].in == 2, "max_swaps keeps the largest gains");
    return v.finish();
}
