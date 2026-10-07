// src/ds41/tests/host_experts_test.cpp - the RAM tier: its plan (profile order, skipping VRAM experts, within a byte
// budget: an expert that does not fit is skipped and smaller ones after it still enter, as upstream's resident budget),
// the compact arena of experts of different sizes (SAGE 1.59bpw), and the automatic budget (never negative, below
// MemAvailable). The CPU kernel pointers are covered by the engine run (byte-identical dumps with and without the tier).
#include "strata/ds41/host_experts.hpp"

#include "fake_pack.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace strata::ds41;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what.c_str()); }
}

int main() {
    {
        const int L = 2, E = 4;
        // rank: (1,3) (0,0) (0,2) (1,1) (0,1) ...; (0,0) and (1,1) are in VRAM
        const std::vector<std::pair<int, int>> ranked = {{1, 3}, {0, 0}, {0, 2}, {1, 1}, {0, 1}, {1, 0}, {0, 3}, {1, 2}};
        std::vector<int32_t> res(L * E, -1);
        res[0 * E + 0] = 0;
        res[1 * E + 1] = 1;
        const std::vector<uint64_t> same(L * E, 10);
        auto p = plan_ram_tier(ranked, res, E, same, 30);
        check(p.size() == 3 && p[0] == std::make_pair(1, 3) && p[1] == std::make_pair(0, 2) &&
                  p[2] == std::make_pair(0, 1),
              "three experts' bytes: the hottest non-VRAM experts in rank order");
        check(plan_ram_tier(ranked, res, E, same, 39).size() == 3, "a partial expert does not enter");
        check(plan_ram_tier(ranked, res, E, same, 1000).size() == 6, "every non-VRAM expert when the budget is large");
        check(plan_ram_tier(ranked, res, E, same, 0).empty(), "no budget: nothing");
        // sizes differ: (1,3) 10, (0,2) 25 (does not fit after it in 30), (0,1) 10, (1,0) 10
        std::vector<uint64_t> sz(L * E, 10);
        sz[0 * E + 2] = 25;
        p = plan_ram_tier(ranked, res, E, sz, 30);
        check(p.size() == 3 && p[0] == std::make_pair(1, 3) && p[1] == std::make_pair(0, 1) &&
                  p[2] == std::make_pair(1, 0),
              "a large expert that does not fit is skipped; smaller ones after it enter");
    }
    {
        // the compact arena on a pack whose experts have 1, 2 or 3 units of 4 KiB
        const std::string dir = "ds41_fake_pack_mixed";
        ds41test::write_fake_pack(dir, true);
        Pack pack(dir);
        pack.map_experts();
        std::vector<std::pair<int, int>> ranked;
        for (int i = 0; i < 40; ++i) ranked.push_back({i % ds41test::L, (i * 7) % ds41test::E});
        const std::vector<int32_t> res((size_t) ds41test::L * ds41test::E, -1);
        uint64_t want = 0;
        for (int i = 0; i < 20; ++i) want += pack.expert(ranked[i].first, ranked[i].second).bytes;
        // the fill reads the slots with O_DIRECT where the file system allows it (as the expert stream does), else it
        // copies from the mapped pack (DS41_FILL_DIRECT=0 forces that path); both must give the same bytes
        for (const char* direct : {"0", "1"}) {
            setenv("DS41_FILL_DIRECT", direct, 1);
            HostExperts h(pack, ranked, res, want, {}, 2);
            const int fd = open((dir + "/experts.bin").c_str(), O_RDONLY | O_DIRECT);
            const bool can = fd >= 0;
            if (fd >= 0) close(fd);
            check(h.filled_direct() == (direct[0] == '1' && can),
                  std::string("DS41_FILL_DIRECT=") + direct + ": O_DIRECT fill " + (h.filled_direct() ? "on" : "off"));
            bool same = h.slots() == 20;
            for (int i = 0; i < h.slots(); ++i) {
                const ExpertSlot& x = pack.expert(ranked[i].first, ranked[i].second);
                same &= std::memcmp(h.slot_ptr(h.slot_of(ranked[i].first, ranked[i].second)),
                                    pack.expert_base() + x.offset, x.bytes) == 0;
            }
            check(same, std::string("DS41_FILL_DIRECT=") + direct + ": every slot holds its expert's bytes");
        }
        unsetenv("DS41_FILL_DIRECT");
        // DS41_RAM_HUGEPAGES=1: the arena starts on a 2 MiB boundary (transparent huge pages), same bytes
        {
            setenv("DS41_RAM_HUGEPAGES", "1", 1);
            HostExperts h(pack, ranked, res, want, {}, 2);
            check(h.huge_pages() && (uintptr_t) h.slot_ptr(0) % (2u << 20) == 0,
                  "DS41_RAM_HUGEPAGES=1: the arena is 2 MiB aligned and asks for huge pages");
            bool same = h.slots() == 20;
            for (int i = 0; i < h.slots(); ++i) {
                const ExpertSlot& x = pack.expert(ranked[i].first, ranked[i].second);
                same &= std::memcmp(h.slot_ptr(h.slot_of(ranked[i].first, ranked[i].second)),
                                    pack.expert_base() + x.offset, x.bytes) == 0;
            }
            check(same, "DS41_RAM_HUGEPAGES=1: every slot holds its expert's bytes");
            setenv("DS41_RAM_HUGEPAGES", "yes", 1);
            bool threw = false;
            try {
                HostExperts bad(pack, ranked, res, want, {}, 2);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            check(threw, "DS41_RAM_HUGEPAGES other than 0 or 1 is refused");
            unsetenv("DS41_RAM_HUGEPAGES");
        }
        HostExperts host(pack, ranked, res, want, {}, 2);
        check(!host.huge_pages(), "huge pages are off by default");
        check(host.slots() == 20, "the budget of the first 20 experts' bytes holds 20 of them, got " +
                                      std::to_string(host.slots()));
        check(host.arena_bytes() == want, "the arena is the sum of the experts' bytes (no slot of the largest size)");
        bool bytes_ok = true, layout_ok = true;
        uint64_t at = 0;
        for (int i = 0; i < host.slots(); ++i) {
            const auto [l, e] = ranked[i];
            const ExpertSlot& x = pack.expert(l, e);
            const int s = host.slot_of(l, e);
            layout_ok &= s == i && host.slot_ptr(s) == host.slot_ptr(0) + at && host.slot_capacity(s) == x.bytes;
            bytes_ok &= s >= 0 && std::memcmp(host.slot_ptr(s), pack.expert_base() + x.offset, x.bytes) == 0;
            at += x.bytes;
        }
        check(layout_ok, "slots follow each other, each as large as its first expert");
        check(bytes_ok, "every slot holds its expert's bytes");
        check(host.max_slot_bytes() == 3 * ds41test::kExpertBytes, "the largest slot");
        // assign: an expert that fits the slot's capacity may take it; a larger one may not
        int small = -1, large = -1;
        for (int s = 0; s < host.slots(); ++s) {
            if (host.slot_capacity(s) == ds41test::kExpertBytes) small = s;
            if (host.slot_capacity(s) == 3 * ds41test::kExpertBytes) large = s;
        }
        check(small >= 0 && large >= 0, "the plan holds a small and a large expert");
        if (small >= 0 && large >= 0) {
            // (l, e) with 1 unit and another with 3 units, outside the tier
            std::pair<int, int> one{-1, -1}, three{-1, -1};
            for (int l = 0; l < ds41test::L && (one.first < 0 || three.first < 0); ++l)
                for (int e = 0; e < ds41test::E; ++e) {
                    if (host.slot_of(l, e) >= 0) continue;
                    if (ds41test::fake_mult(l, e) == 1 && one.first < 0) one = {l, e};
                    if (ds41test::fake_mult(l, e) == 3 && three.first < 0) three = {l, e};
                }
            bool threw = false;
            try { host.assign(small, three.first, three.second); } catch (const std::exception&) { threw = true; }
            check(threw, "a larger expert cannot take a small slot");
            const auto [ol, oe] = ranked[large];
            host.point_to_file(ol, oe);
            host.assign(large, one.first, one.second);
            check(host.slot_of(one.first, one.second) == large && host.slot_of(ol, oe) == -1,
                  "a smaller expert takes a large slot; the old one leaves the tier");
            check(host.slot_capacity(large) == 3 * ds41test::kExpertBytes, "the slot keeps its capacity");
        }
    }

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
