// src/ds41/tests/vram_experts_test.cu - the host logic of the VRAM expert tier: the STRP profile reader and the
// adaptive swap plan (upstream's rule). The slot copies and K10 are covered by the engine run.
#include "strata/ds41/vram_experts.hpp"

#include "bench_util.hpp"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

using namespace ds41test;
using strata::ds41::ExpertSwap;
using strata::ds41::plan_expert_swaps;
using strata::ds41::read_expert_profile;
using strata::ds41::uma_free_bytes;
using strata::ds41::ram_tier_headroom;
using strata::ds41::uma_headroom_bytes;

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

    // ---- free device memory on unified memory (upstream device_free_bytes): the OS's MemAvailable less the headroom
    // counts when it is more than what cudaMemGetInfo reports (the DGX Spark reported 23.8 GiB free with 113 GiB
    // available, the rest a page cache the OS gives back)
    const uint64_t G = 1ull << 30;
    v.check(uma_free_bytes(24 * G, 113 * G, 6 * G) == 107 * G, "UMA: MemAvailable less the headroom");
    v.check(uma_free_bytes(24 * G, 20 * G, 6 * G) == 24 * G, "UMA: never less than cudaMemGetInfo");
    v.check(uma_free_bytes(1 * G, 5 * G, 6 * G) == 1 * G, "UMA: MemAvailable under the headroom adds nothing");
    // the RAM tier's headroom: on unified memory it also keeps the GPU's headroom free, or device_free_bytes() finds
    // almost nothing after the tier (DGX Spark, DSpark on: 713 MiB, too little for a 16-token prefill pass)
    v.check(ram_tier_headroom(true, false, 6 * G) == 8 * G, "RAM tier headroom: adaptive, discrete GPU");
    v.check(ram_tier_headroom(false, false, 6 * G) == 24 * G, "RAM tier headroom: static, discrete GPU");
    v.check(ram_tier_headroom(true, true, 6 * G) == 14 * G, "RAM tier headroom: adaptive, unified memory");
    v.check(ram_tier_headroom(false, true, 2 * G) == 26 * G, "RAM tier headroom: static, unified memory");
    if (std::getenv("STRATA_UMA_HEADROOM_GIB") == nullptr)
        v.check(uma_headroom_bytes() == 6 * G, "UMA headroom: 6 GiB by default");

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
    // the same plan through a fit test that accepts everything (experts of one size)
    auto all = [](int, int, int) { return true; };
    v.check(plan_expert_swaps(u, res, L, E, 96, all).size() == 4, "a fit test that accepts all: the same plan");

    // ---- experts of different sizes (SAGE 1.59bpw): `in` must fit the slot of `out`
    // layer 0: resident e0 (usage 0, small slot), e1 (0.5, large slot); missing e2 (9, large), e3 (8, small)
    std::vector<int32_t> r2(L * E, -1);
    std::vector<float> u2(L * E, 0.0f);
    r2[0] = 0; r2[1] = 1;
    u2[0] = 0; u2[1] = 0.5f; u2[2] = 9; u2[3] = 8;
    auto large = [](int l, int e) { return l == 0 && (e == 1 || e == 2); };
    auto fits = [&](int l, int in, int out) { return !large(l, in) || large(l, out); };
    s = plan_expert_swaps(u2, r2, L, E, 96, fits);
    // e2 (large) cannot take e0's small slot: it takes e1's; e3 (small) then takes e0's
    v.check(s.size() == 2, "two swaps with sizes, got " + std::to_string(s.size()));
    if (s.size() == 2) {
        v.check(s[0].in == 2 && s[0].out == 1 && s[0].gain == 8.5f, "the large candidate takes the large slot");
        v.check(s[1].in == 3 && s[1].out == 0 && s[1].gain == 8.0f, "the small candidate takes the small slot");
    }
    // only a small slot left: the large candidate is skipped, the small one after it still swaps
    r2[1] = -1;
    u2[1] = 0;
    s = plan_expert_swaps(u2, r2, L, E, 96, fits);
    v.check(s.size() == 1 && s[0].in == 3 && s[0].out == 0, "a candidate that fits no slot is skipped");
    return v.finish();
}
