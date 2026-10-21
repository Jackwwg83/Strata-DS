// src/ds41/tests/host_experts_test.cpp - the RAM tier: its plan (profile order, skipping VRAM experts, within a byte
// budget: an expert that does not fit is skipped and smaller ones after it still enter, as upstream's resident budget),
// the compact arena of experts of different sizes (SAGE 1.59bpw), and the automatic budget (never negative, below
// MemAvailable). The CPU kernel pointers are covered by the engine run (byte-identical dumps with and without the tier).
#include "strata/ds41/host_experts.hpp"

#include "fake_pack.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <vector>

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
        // host_alloc (unified memory): the arena from cudaHostAlloc, which the DGX Spark's GPU reads at 210 GB/s
        // against 166 GB/s for registered anonymous memory; the same bytes, mapped, read with O_DIRECT
        {
            int n_dev = 0;
            if (cudaGetDeviceCount(&n_dev) == cudaSuccess && n_dev > 0) {
                HostExperts h(pack, ranked, res, want, {}, 2, true);
                bool same = h.slots() == 20 && h.host_allocated() && h.mapped() && h.locked();
                for (int i = 0; i < h.slots(); ++i) {
                    const ExpertSlot& x = pack.expert(ranked[i].first, ranked[i].second);
                    same &= std::memcmp(h.slot_ptr(h.slot_of(ranked[i].first, ranked[i].second)),
                                        pack.expert_base() + x.offset, x.bytes) == 0;
                }
                check(same, "host_alloc: a mapped host arena; every slot holds its expert's bytes");
                // an integrated GPU (DGX Spark): cuMemCreate on the host NUMA node, which its GPU reads at 233 GB/s
                // against 220 for cudaHostAlloc; the GPU reads the slots' bytes through the device address
                int integrated = 0;
                cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0);
                if (integrated) {
                    check(h.numa_allocated(), "host_alloc on unified memory: the arena from cuMemCreate (host NUMA)");
                    const ExpertSlot& x = pack.expert(ranked[3].first, ranked[3].second);
                    std::vector<uint8_t> back(x.bytes);
                    const uint8_t* dev = h.device_address(h.slot_ptr(h.slot_of(ranked[3].first, ranked[3].second)));
                    const bool copied = dev && cudaMemcpy(back.data(), dev, x.bytes, cudaMemcpyDefault) == cudaSuccess;
                    check(copied && std::memcmp(back.data(), pack.expert_base() + x.offset, x.bytes) == 0,
                          "host_alloc on unified memory: the GPU reads the slot's bytes");
                }
            } else {
                cudaGetLastError();
            }
        }
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

    // one admit of several experts, each read in several parts (4 KiB parts here, 2 MiB in use): the parts go to the
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
        const size_t part_bytes = detail::admit_part_bytes();
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
            check(h.slot_bytes(layer, ids[0]) == nullptr, tag + "slot_bytes: null for an expert in no slot");
            const int n = h.admit(layer, ids, 3, ok);
            // during the step, before end_step publishes them: the admitted experts' bytes are in their new slots
            bool found = true;
            for (int k = 0; k < 3; ++k) {
                const uint8_t* b = h.slot_bytes(layer, ids[k]);
                const ExpertSlot& x = pack.expert(layer, ids[k]);
                found &= b != nullptr && std::memcmp(b, pack.expert_base() + x.offset, x.bytes) == 0;
            }
            check(found, tag + "slot_bytes: an expert admitted this step is found in its new slot");
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
        detail::admit_part_bytes() = part_bytes;
    }

    // prefetch: reads started before admit() asks (the router lookahead's guesses for the next layer). admit() takes an
    // expert read that way without reading it again; one that no admit() took leaves at end_step() (its slot free); a
    // failed read gives its slot back and admit() reads the expert itself. Between end_step() and open_prefetch()
    // prefetch() starts nothing (the slots change between steps).
    {
        const std::string tag = "prefetch: ";
        const std::string dir = "ds41_fake_pack_prefetch";
        ds41test::write_fake_pack(dir, true, 4096);
        Pack pack(dir);
        pack.map_experts();
        std::vector<std::pair<int, int>> ranked;
        for (int i = 0; i < 40; ++i) ranked.push_back({i % ds41test::L, (i * 7) % ds41test::E});
        const std::vector<int32_t> res((size_t) ds41test::L * ds41test::E, -1);
        uint64_t budget = 0;
        for (int i = 0; i < 24; ++i) budget += pack.expert(ranked[i].first, ranked[i].second).bytes;
        HostExperts h(pack, ranked, res, budget, {}, 2);
        h.enable_adapt(3);
        const size_t part_bytes = detail::admit_part_bytes();
        detail::admit_part_bytes() = 4096;   // the prefetch reads go in parts too (2 MiB in use)
        const int free0 = h.free_slots();
        auto outside = [&](int l) {   // experts of layer l in no tier
            std::vector<int32_t> v;
            for (int e = 0; e < ds41test::E; ++e) {
                bool ranked_pair = false;
                for (auto& p : ranked) ranked_pair |= p == std::make_pair(l, e);
                if (!ranked_pair) v.push_back(e);
            }
            return v;
        };
        auto bytes_ok = [&](int l, int e) {
            const int s = h.slot_of(l, e);
            const ExpertSlot& x = pack.expert(l, e);
            return s >= 0 && std::memcmp(h.slot_ptr(s), pack.expert_base() + x.offset, x.bytes) == 0;
        };
        const int layer = 3;
        const auto out = outside(layer);
        check(out.size() >= 4, tag + "four experts of layer 3 outside the tier");
        const int32_t ids[3] = {out[0], out[1], out[2]};
        const int32_t nxt = outside(layer + 1)[0];   // a guess for the next layer, which no admit() asks for
        check(h.prefetch(layer, ids, 1) == 0, tag + "nothing starts before open_prefetch()");
        h.open_prefetch();
        check(h.prefetch(layer, ids, 1) == 1 && h.prefetch(layer + 1, &nxt, 1) == 1, tag + "two reads start");
        check(h.prefetch(layer, ids, 1) == 0, tag + "an expert on its way is not read twice");
        const int32_t in_tier = ranked[0].second;
        check(h.prefetch(ranked[0].first, &in_tier, 1) == 0, tag + "an expert in the tier is not read");
        bool ok[2] = {};
        check(h.admit(layer, &ids[0], 1, ok) == 1 && ok[0] && h.prefetch_used_total() == 1,
              tag + "admit() takes the prefetched expert");
        check(h.admit(layer, &ids[2], 1, ok + 1) == 1 && ok[1] && h.prefetch_used_total() == 1,
              tag + "admit() reads an expert nobody prefetched");
        check(h.slot_of(layer, ids[0]) == -1 && h.slot_of(layer + 1, nxt) == -1,
              tag + "the RAM table does not change during the step");
        std::vector<int32_t> routes((size_t) ds41test::L * 6, -1);
        routes[(size_t) layer * 6 + 0] = ids[0];
        routes[(size_t) layer * 6 + 1] = ids[2];
        h.end_step(routes.data(), 6);
        check(bytes_ok(layer, ids[0]) && bytes_ok(layer, ids[2]), tag + "both admitted experts are published with their bytes");
        check(h.slot_of(layer + 1, nxt) == -1, tag + "the unused guess does not enter the tier");
        check(h.prefetched_total() == 2 && h.admitted_total() == 2, tag + "the totals");
        check(h.free_slots() == free0, tag + "end_step keeps the reserve free");
        const int32_t late = out[3];
        check(h.prefetch(layer, &late, 1) == 0, tag + "after end_step() nothing starts until open_prefetch()");
        h.open_prefetch();
        detail::admit_fault() = true;   // the prefetch read takes the fault: admit() waits for it, then reads itself
        check(h.prefetch(layer, &late, 1) == 1, tag + "a read starts");
        bool ok2 = false;
        check(h.admit(layer, &late, 1, &ok2) == 1 && ok2 && h.prefetch_used_total() == 1,
              tag + "a failed prefetch: admit() reads the expert itself");
        detail::admit_fault() = false;
        routes[(size_t) layer * 6 + 2] = late;
        h.end_step(routes.data(), 6);
        check(bytes_ok(layer, late) && h.free_slots() == free0, tag + "its bytes, and the failed read's slot is free again");
        // two guesses for layer 5, admit() asks for one: the other was wrong, and its parts not yet read are dropped
        h.open_prefetch();
        const auto out5 = outside(5);
        const int32_t g[2] = {out5[0], out5[1]};
        check(h.prefetch(5, g, 2) == 2, tag + "two guesses for layer 5");
        bool ok3 = false;
        check(h.admit(5, &g[0], 1, &ok3) == 1 && ok3, tag + "admit() takes one");
        std::vector<int32_t> r5((size_t) ds41test::L * 6, -1);
        r5[(size_t) 5 * 6] = g[0];
        h.end_step(r5.data(), 6);
        check(bytes_ok(5, g[0]) && h.slot_of(5, g[1]) == -1 && h.free_slots() == free0,
              tag + "the wrong guess, dropped or read in full, does not enter the tier");
        detail::admit_part_bytes() = part_bytes;
    }

    // pinning that fails while the arena is large (memory short at start): the tier drops its lowest-ranked slots, a
    // step at a time, until the GPU can map it, instead of staying unpinned (paged out, CPU only)
    {
        const std::string dir = "ds41_fake_pack_pin";
        ds41test::write_fake_pack(dir, true, 4096);
        Pack pack(dir);
        pack.map_experts();
        std::vector<std::pair<int, int>> ranked;
        for (int i = 0; i < 40; ++i) ranked.push_back({i % ds41test::L, (i * 7) % ds41test::E});
        const std::vector<int32_t> res((size_t) ds41test::L * ds41test::E, -1);
        uint64_t budget = 0;
        for (int i = 0; i < 32; ++i) budget += pack.expert(ranked[i].first, ranked[i].second).bytes;
        HostExperts full(pack, ranked, res, budget, {}, 2);
        if (!full.mapped()) {
            std::printf("pin fallback: skipped (no GPU mapping here)\n");
        } else {
            const size_t limit = full.arena_bytes() * 6 / 10;
            detail::register_limit() = limit;
            HostExperts h(pack, ranked, res, budget, {}, 2);
            detail::register_limit() = 0;
            check(h.mapped() && h.locked(), "pin fallback: the smaller tier is pinned and mapped");
            check(h.arena_bytes() <= limit && h.arena_bytes() >= full.arena_bytes() / 2 && h.slots() < full.slots(),
                  "pin fallback: it dropped slots until the pin fit, not more than half the arena");
            bool prefix = true;
            for (const auto& [l, e] : ranked) {
                const int s = full.slot_of(l, e);
                if (s < 0) continue;
                const ExpertSlot& x = pack.expert(l, e);
                if (s < h.slots())
                    prefix &= h.slot_of(l, e) == s &&
                              std::memcmp(h.slot_ptr(s), pack.expert_base() + x.offset, x.bytes) == 0;
                else
                    prefix &= h.slot_of(l, e) == -1;
            }
            check(prefix, "pin fallback: the highest-ranked slots stay with their bytes; the dropped ones use the file");
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
    {
        // DSpark's experts (pack.py --mtp-experts): mtp_experts.txt in the layout of experts.txt, the layer field the
        // stage; load_mtp_experts reads every slot into RAM
        const std::string dir = "ds41_fake_pack_mtp";
        ds41test::write_fake_pack(dir);
        {
            Pack plain(dir);
            check(plain.mtp_stages() == 0 && plain.mtp_experts() == 0, "a pack without mtp_experts.txt has no DSpark experts");
        }
        const int S = 3, E = 128;
        const uint64_t slot = 8192;
        {
            std::ofstream f(dir + "/mtp_experts.txt");
            f << "# header\n";
            const char* comp[12] = {"w1.trellis", "w1.suh", "w1.svh", "w1.mul1", "w3.trellis", "w3.suh",
                                    "w3.svh",     "w3.mul1", "w2.trellis", "w2.suh", "w2.svh", "w2.mul1"};
            for (int st = 0; st < S; ++st)
                for (int e = 0; e < E; ++e) {
                    f << st << " " << e << " " << ((uint64_t) st * E + e) * slot << " " << slot << " 4 4 4";
                    for (int c = 0; c < 12; ++c) f << " " << comp[c] << ":" << c * 512 << ":" << 512;
                    f << "\n";
                }
            std::vector<uint8_t> b((size_t) S * E * slot);
            for (size_t i = 0; i < b.size(); ++i) b[i] = (uint8_t) (i * 2654435761u >> 13);
            std::ofstream(dir + "/mtp_experts.bin", std::ios::binary).write((const char*) b.data(), (std::streamsize) b.size());
            std::ofstream(dir + "/pack_info.txt") << "layers " << ds41test::L << "\nexperts " << ds41test::E
                                                  << "\nmtp_stages 3\nmtp_experts 128\nfinished 1\n";
        }
        Pack pack(dir);
        check(pack.mtp_stages() == S && pack.mtp_experts() == E, "pack_info.txt names 3 stages of 128 DSpark experts");
        check(pack.mtp_expert(2, 127).offset == (uint64_t) (2 * E + 127) * slot && pack.mtp_expert(1, 5).bits[0] == 4,
              "mtp_expert reads mtp_experts.txt");
        check(pack.load_mtp_experts() == (uint64_t) S * E * slot, "load_mtp_experts reads the whole file");
        bool same = true;
        for (int st = 0; st < S; ++st)
            for (int e = 0; e < E; e += 17) {
                const auto& x = pack.mtp_expert(st, e);
                for (uint64_t i = 0; i < x.bytes; i += 509)
                    same &= pack.mtp_expert_base()[x.offset + i] == (uint8_t) ((x.offset + i) * 2654435761u >> 13);
            }
        check(same, "the loaded experts hold the file's bytes");
        std::remove((dir + "/mtp_experts.txt").c_str());
        bool threw = false;
        try {
            Pack missing(dir);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "pack_info.txt names DSpark experts that mtp_experts.txt does not list: refused");
    }
    std::printf("RESULT %s\n", failures ? "fail" : "pass");
    return failures ? 1 : 0;
}
