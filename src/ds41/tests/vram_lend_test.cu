// src/ds41/tests/vram_lend_test.cu - VramExperts::lend / restore (prefill borrows cache slots): the lent slots leave
// the residency table, the caller may overwrite them, and restore() brings back every byte and table entry.
// Runs on fake packs: full 40 x 384 geometry, experts of random bytes, no dense weights. One pack has 4 KiB experts;
// the other has experts of 4, 8 and 12 KiB (as SAGE 1.59bpw's differ): the slots are compact, and prefill borrows
// bytes (lend_bytes), which the last slots cover.
#include "strata/ds41/vram_experts.hpp"

#include "bench_util.hpp"
#include "fake_pack.hpp"

#include <algorithm>
#include <cstring>

using namespace ds41test;
namespace sd = strata::ds41;

namespace {

void mixed_sizes(Verdict& v) {
    const std::string dir = "/tmp/ds41_fake_pack_mixed", prof = "/tmp/ds41_fake_profile_mixed.bin";
    write_fake_pack(dir, true);
    std::vector<std::pair<int, int>> ranked;
    for (int i = 0; i < 300; ++i) ranked.push_back({(i * 7) % L, (i * 13) % E});
    write_profile(prof, ranked);
    sd::Pack pack(dir);
    pack.map_experts();
    sd::VramExperts::Adapt ad;
    ad.every = 0;
    constexpr int kSlots = 100;
    sd::VramExperts vram(pack, prof, kSlots, 0, ad);
    uint64_t sum = 0, mx = 0;
    for (int i = 0; i < kSlots; ++i) {
        const uint64_t b = pack.expert(ranked[i].first, ranked[i].second).bytes;
        sum += b;
        mx = std::max(mx, b);
    }
    v.check(vram.arena_bytes() == sum, "mixed: the arena is the sum of the experts' bytes");
    v.check(vram.max_slot_bytes() == mx, "mixed: the largest slot");
    bool layout_ok = true;
    for (int s = 0; s + 1 < kSlots; ++s)
        layout_ok &= (const uint8_t*) vram.desc(s + 1).w1.trellis ==
                     (const uint8_t*) vram.desc(s).w1.trellis + pack.expert(ranked[s].first, ranked[s].second).bytes;
    v.check(layout_ok, "mixed: the slots follow each other");
    const auto before = vram.res_host();
    // borrow 10 KiB more than the last 7 slots hold: the last 8 (or more) slots are lent
    uint64_t tail7 = 0;
    for (int s = kSlots - 7; s < kSlots; ++s) tail7 += pack.expert(ranked[s].first, ranked[s].second).bytes;
    const size_t want = tail7 + 10 * 1024;
    uint8_t* lent = vram.lend_bytes(want);
    const int n = vram.lent();
    v.check(n >= 8, "mixed: the lent slots cover the bytes asked for");
    v.check(vram.tail_bytes(n) >= want && vram.tail_bytes(n - 1) < want, "mixed: no more slots than needed");
    v.check(lent == (const uint8_t*) vram.desc(kSlots - n).w1.trellis, "mixed: lend returns the first lent slot");
    ck(cudaMemset(lent, 0xCD, want), "overwrite lent bytes");
    vram.restore();
    v.check(vram.res_host() == before, "mixed: restore brings back the table");
    bool bytes_ok = true;
    for (int s = 0; s < kSlots; ++s) {
        const auto& x = pack.expert(ranked[s].first, ranked[s].second);
        std::vector<uint8_t> got(x.bytes);
        ck(cudaMemcpy(got.data(), vram.desc(s).w1.trellis, x.bytes, cudaMemcpyDeviceToHost), "read slot");
        bytes_ok &= std::memcmp(got.data(), pack.expert_base() + x.offset, x.bytes) == 0;
    }
    v.check(bytes_ok, "mixed: every slot holds its expert's bytes again");
}

}  // namespace

int main() {
    require_gpu();
    Verdict v;
    mixed_sizes(v);
    const std::string dir = "/tmp/ds41_fake_pack", prof = "/tmp/ds41_fake_profile.bin";
    write_fake_pack(dir);
    std::vector<std::pair<int, int>> ranked;
    for (int i = 0; i < 300; ++i) ranked.push_back({(i * 7) % L, (i * 13) % E});
    write_profile(prof, ranked);
    sd::Pack pack(dir);
    pack.map_experts();
    sd::VramExperts::Adapt ad;
    ad.every = 0;
    constexpr int kSlots = 100, kLend = 30;
    sd::VramExperts vram(pack, prof, kSlots, 0, ad);
    v.check(vram.slots() == kSlots, "slot count");
    const auto before = vram.res_host();

    uint8_t* lent = vram.lend(kLend);
    v.check(vram.lent() == kLend, "lent count");
    const auto during = vram.res_host();
    int gone = 0, kept = 0;
    bool table_ok = true;
    for (size_t i = 0; i < before.size(); ++i) {
        if (before[i] >= kSlots - kLend) { ++gone; table_ok &= during[i] == -1; }
        else if (before[i] >= 0) { ++kept; table_ok &= during[i] == before[i]; }
        else table_ok &= during[i] == -1;
    }
    v.check(table_ok && gone == kLend && kept == kSlots - kLend, "lend: the last slots leave the table, the rest stay");
    // the lent memory is the last kLend slots: slot kSlots - kLend starts at `lent`
    v.check((const uint8_t*) vram.desc(kSlots - kLend).w1.trellis == lent, "lend returns the first lent slot");
    ck(cudaMemset(lent, 0xAB, (size_t) kLend * vram.max_slot_bytes()), "overwrite lent slots");
    v.check(vram.tail_bytes(kLend) == (size_t) kLend * vram.max_slot_bytes(), "one size: the lent bytes");
    bool threw = false;
    try { vram.lend(1); } catch (const std::exception&) { threw = true; }
    v.check(threw, "a second lend is refused");

    vram.restore();
    v.check(vram.lent() == 0 && vram.res_host() == before, "restore: the table is as before");
    bool bytes_ok = true;
    for (size_t i = 0; i < before.size(); ++i) {
        const int s = before[i];
        if (s < 0) continue;
        std::vector<uint8_t> got(kExpertBytes);
        ck(cudaMemcpy(got.data(), vram.desc(s).w1.trellis, kExpertBytes, cudaMemcpyDeviceToHost), "read slot");
        const auto& x = pack.expert((int) (i / E), (int) (i % E));
        bytes_ok &= std::memcmp(got.data(), pack.expert_base() + x.offset, kExpertBytes) == 0;
    }
    v.check(bytes_ok, "restore: every slot holds its expert's bytes again");
    return v.finish();
}
