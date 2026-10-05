// src/ds41/tests/host_experts_test.cpp - the RAM tier's plan (profile order, skipping VRAM experts, at most n slots)
// and the automatic budget (never negative, below MemAvailable). The copies and the CPU pointers are covered by the
// engine run (byte-identical dumps with and without the tier).
#include "strata/ds41/host_experts.hpp"

#include <cstdio>
#include <fstream>
#include <string>

using namespace strata::ds41;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what.c_str()); }
}

int main() {
    const int L = 2, E = 4;
    // rank: (1,3) (0,0) (0,2) (1,1) (0,1) ...; (0,0) and (1,1) are in VRAM
    const std::vector<std::pair<int, int>> ranked = {{1, 3}, {0, 0}, {0, 2}, {1, 1}, {0, 1}, {1, 0}, {0, 3}, {1, 2}};
    std::vector<int32_t> res(L * E, -1);
    res[0 * E + 0] = 0;
    res[1 * E + 1] = 1;
    auto p = plan_ram_tier(ranked, res, E, 3);
    check(p.size() == 3 && p[0] == std::make_pair(1, 3) && p[1] == std::make_pair(0, 2) && p[2] == std::make_pair(0, 1),
          "three slots: the hottest non-VRAM experts in rank order");
    p = plan_ram_tier(ranked, res, E, 100);
    check(p.size() == 6, "every non-VRAM expert when the budget is large");
    check(plan_ram_tier(ranked, res, E, 0).empty(), "no slots: nothing");

    const size_t b = auto_ram_budget(4ull << 30);
    uint64_t avail_kb = 0;
    {
        std::ifstream f("/proc/meminfo");
        std::string k;
        uint64_t v;
        while (f >> k >> v) { if (k == "MemAvailable:") avail_kb = v; std::getline(f, k); }
    }
    check(b <= avail_kb * 1024, "the budget is below MemAvailable");
    std::printf("auto budget %.1f GiB (MemAvailable %.1f GiB)\n", b / 1073741824.0, avail_kb / 1048576.0);
    std::printf("RESULT %s\n", failures ? "fail" : "pass");
    return failures ? 1 : 0;
}
