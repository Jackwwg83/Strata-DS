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
        HostExperts host(pack, ranked, res, want, {}, 2);
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

    // the adaptive tier, on a pack whose experts start off 4 KiB boundaries (copied from the map) and on one whose
    // experts all start on one (read with O_DIRECT where the file system allows it)
    for (const uint64_t gap : {(uint64_t) 256, (uint64_t) 4096}) {
        const std::string tag = "gap " + std::to_string(gap) + ": ";
        const std::string dir = "ds41_fake_pack_adapt_" + std::to_string(gap);
        ds41test::write_fake_pack(dir, true, gap);
        Pack pack(dir);
        pack.map_experts();
        std::vector<std::pair<int, int>> ranked;
        for (int i = 0; i < 40; ++i) ranked.push_back({i % ds41test::L, (i * 7) % ds41test::E});
        const std::vector<int32_t> res((size_t) ds41test::L * ds41test::E, -1);
        uint64_t budget = 0;
        for (int i = 0; i < 24; ++i) budget += pack.expert(ranked[i].first, ranked[i].second).bytes;
        HostExperts h(pack, ranked, res, budget, {}, 2);
        const uint64_t unit = ds41test::kExpertBytes;
        auto cap_free = [&](uint64_t cap) {   // slots of this capacity that no expert's table entry points at
            std::vector<int> held(h.slots(), 0);
            for (int l = 0; l < ds41test::L; ++l)
                for (int e = 0; e < ds41test::E; ++e)
                    if (h.slot_of(l, e) >= 0) held[h.slot_of(l, e)] = 1;
            int n = 0;
            for (int s = 0; s < h.slots(); ++s) n += h.slot_capacity(s) == cap && !held[s];
            return n;
        };
        auto bytes_ok = [&](int l, int e) {
            const int s = h.slot_of(l, e);
            const ExpertSlot& x = pack.expert(l, e);
            return s >= 0 && std::memcmp(h.slot_ptr(s), pack.expert_base() + x.offset, x.bytes) == 0;
        };
        // the last-ranked slot of each capacity is freed
        int last[4] = {-1, -1, -1, -1};
        for (int s = 0; s < h.slots(); ++s) last[h.slot_capacity(s) / unit] = s;
        check(last[1] >= 0 && last[2] >= 0 && last[3] >= 0, tag + "the tier holds experts of 1, 2 and 3 units");
        h.lock(0);   // a VRAM swap in flight: enabling now would free slots it uses
        bool refused = false;
        try { h.enable_adapt(1); } catch (const std::logic_error&) { refused = true; }
        check(refused && h.reserve() == 0, tag + "enable_adapt refuses while a swap holds a slot");
        h.unlock(0);
        h.enable_adapt(1);
        check(h.free_slots() == 3, tag + "one free slot per capacity, got " + std::to_string(h.free_slots()));
        for (int m = 1; m <= 3; ++m)
            check(h.slot_of(ranked[last[m]].first, ranked[last[m]].second) == -1 && cap_free(m * unit) == 1,
                  tag + "the lowest-ranked expert of " + std::to_string(m) + " units leaves its slot");
        // experts outside the tier, by size
        std::vector<std::pair<int, int>> out[4];
        for (int l = 0; l < ds41test::L; ++l)
            for (int e = 0; e < ds41test::E; ++e) {
                bool ranked_pair = false;
                for (auto& p : ranked) ranked_pair |= p == std::make_pair(l, e);
                if (!ranked_pair && out[ds41test::fake_mult(l, e)].size() < 4) out[ds41test::fake_mult(l, e)].push_back({l, e});
            }
        auto admit1 = [&](std::pair<int, int> x) {
            const int32_t id = x.second;
            bool ok = false;
            const int n = h.admit(x.first, &id, 1, &ok);
            return n == 1 && ok;
        };
        auto routes_of = [&](std::initializer_list<std::pair<int, int>> used) {
            std::vector<int32_t> r((size_t) ds41test::L * 6, -1);
            std::vector<int> fill(ds41test::L, 0);
            for (auto [l, e] : used) r[(size_t) l * 6 + fill[l]++] = e;
            return r;
        };
        const auto a = out[1][0], b = out[1][1], c = out[3][0], d = out[3][1];
        {   // a read that throws: the picked slot is free again and the expert stays in the file
            detail::admit_fault() = true;
            bool threw = false;
            try { admit1(a); } catch (const std::runtime_error&) { threw = true; }
            check(threw && h.free_slots() == 3 && h.slot_of(a.first, a.second) == -1,
                  tag + "a failed admit gives its slot back");
            std::vector<int32_t> none((size_t) ds41test::L * 6, -1);
            check(h.end_step(none.data(), 6) == 0 && h.admitted_total() == 0, tag + "and publishes nothing");
        }
        check(admit1(a), tag + "a 1-unit expert is read into the free 1-unit slot");
        check(h.slot_of(a.first, a.second) == -1, tag + "the RAM table does not change during the step");
        check(admit1(b), tag + "the next 1-unit expert takes the free 2-unit slot (the smallest free one that holds it)");
        check(admit1(c), tag + "a 3-unit expert takes the free 3-unit slot");
        check(!admit1(d), tag + "no free slot holds a second 3-unit expert");
        check(h.free_slots() == 0, tag + "every free slot is taken");
        auto r = routes_of({a, b, c, d});
        int ev = h.end_step(r.data(), 6);
        check(ev == 3 && h.free_slots() == 3, tag + "end_step frees one slot per capacity again, evicted " +
                                                  std::to_string(ev));
        check(bytes_ok(a.first, a.second) && h.slot_capacity(h.slot_of(a.first, a.second)) == unit, tag + "a: published, its bytes");
        check(bytes_ok(b.first, b.second) && h.slot_capacity(h.slot_of(b.first, b.second)) == 2 * unit, tag + "b: published in the 2-unit slot");
        check(bytes_ok(c.first, c.second), tag + "c: published, its bytes");
        check(h.slot_of(d.first, d.second) == -1, tag + "d stays in the file");
        check(h.admitted_total() == 3 && h.evicted_total() == 3, tag + "the totals");
        // least recently used first: the 1-unit experts used in the last step stay; the oldest (ties: lowest-ranked)
        // unused one leaves
        std::vector<int> ones;   // occupied 1-unit slots of the profile's experts, highest slot first
        for (int s = h.slots() - 1; s >= 0; --s)
            if (h.slot_capacity(s) == unit && h.slot_of(ranked[s].first, ranked[s].second) == s) ones.push_back(s);
        check(ones.size() >= 2, tag + "two profile experts of 1 unit remain");
        if (ones.size() >= 2) {
            const auto keep = ranked[ones[0]], next = ranked[ones[1]];
            check(admit1(out[1][2]), tag + "another 1-unit expert");
            r = routes_of({keep, out[1][2]});
            h.end_step(r.data(), 6);
            check(h.slot_of(keep.first, keep.second) == ones[0], tag + "a used expert stays");
            check(h.slot_of(next.first, next.second) == -1, tag + "the least recently used one leaves");
            // a locked slot never leaves
            int victim = -1;
            for (int s = h.slots() - 1; s >= 0 && victim < 0; --s)
                if (h.slot_capacity(s) == unit && h.slot_of(ranked[s].first, ranked[s].second) == s && s != ones[0]) victim = s;
            if (victim >= 0) {
                h.lock(victim);
                check(admit1(out[1][3]), tag + "a fourth 1-unit expert");
                r = routes_of({keep, out[1][3]});
                h.end_step(r.data(), 6);
                check(h.slot_of(ranked[victim].first, ranked[victim].second) == victim, tag + "the locked slot stays");
                h.unlock(victim);
            }
        }
        const int before = h.free_slots();
        h.release(a.first, a.second);
        check(h.slot_of(a.first, a.second) == -1 && h.free_slots() == before + 1, tag + "release frees the slot");
        h.release(d.first, d.second);
        check(h.free_slots() == before + 1, tag + "release of an expert without a slot changes nothing");
    }

    // one admit of several experts, each read in several parts (4 KiB parts here, 1 MiB in use): the parts go to the
    // reader threads, and every slot must hold its expert's bytes; twice, so the readers are reused
    for (const uint64_t gap : {(uint64_t) 256, (uint64_t) 4096}) {
        const std::string tag = "parts, gap " + std::to_string(gap) + ": ";
        const std::string dir = "ds41_fake_pack_parts_" + std::to_string(gap);
        ds41test::write_fake_pack(dir, true, gap);
        Pack pack(dir);
        pack.map_experts();
        std::vector<std::pair<int, int>> ranked;
        for (int i = 0; i < 40; ++i) ranked.push_back({i % ds41test::L, (i * 7) % ds41test::E});
        const std::vector<int32_t> res((size_t) ds41test::L * ds41test::E, -1);
        uint64_t budget = 0;
        for (int i = 0; i < 24; ++i) budget += pack.expert(ranked[i].first, ranked[i].second).bytes;
        HostExperts h(pack, ranked, res, budget, {}, 2);
        h.enable_adapt(3);
        detail::admit_part_bytes() = 4096;
        std::vector<std::pair<int, int>> big;   // 3-unit experts outside the tier, of one layer
        for (int l = 0; l < ds41test::L && big.size() < 3; ++l) {
            big.clear();
            for (int e = 0; e < ds41test::E && big.size() < 3; ++e) {
                bool ranked_pair = false;
                for (auto& p : ranked) ranked_pair |= p == std::make_pair(l, e);
                if (!ranked_pair && ds41test::fake_mult(l, e) == 3) big.push_back({l, e});
            }
        }
        check(big.size() == 3, tag + "three 3-unit experts of one layer outside the tier");
        for (int round = 0; round < 2 && big.size() == 3; ++round) {
            const int layer = big[0].first;
            int32_t ids[3] = {big[0].second, big[1].second, big[2].second};
            bool ok[3] = {};
            const int n = h.admit(layer, ids, 3, ok);
            std::vector<int32_t> routes((size_t) ds41test::L * 6, -1);
            for (int k = 0; k < 3; ++k) routes[(size_t) layer * 6 + k] = ids[k];
            h.end_step(routes.data(), 6);
            bool same = n == 3 && ok[0] && ok[1] && ok[2];
            for (int k = 0; k < 3 && same; ++k) {
                const int s = h.slot_of(layer, ids[k]);
                const ExpertSlot& x = pack.expert(layer, ids[k]);
                same &= s >= 0 && std::memcmp(h.slot_ptr(s), pack.expert_base() + x.offset, x.bytes) == 0;
            }
            check(same, tag + "round " + std::to_string(round) + ": one admit read three experts in parts, bytes equal");
            for (auto& [l, e] : big) h.release(l, e);   // free their slots for the next round
        }
        detail::admit_part_bytes() = 1u << 20;
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
