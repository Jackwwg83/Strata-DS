// src/ds41/tests/vram_lend_test.cu - VramExperts::lend / restore (prefill borrows cache slots): the lent slots leave
// the residency table, the caller may overwrite them, and restore() brings back every byte and table entry.
// Runs on a fake pack: full 40 x 384 geometry, 4 KiB experts of random bytes, no dense weights.
#include "strata/ds41/vram_experts.hpp"

#include "bench_util.hpp"
#include "fake_pack.hpp"

using namespace ds41test;
namespace sd = strata::ds41;

int main() {
    require_gpu();
    Verdict v;
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
    ck(cudaMemset(lent, 0xAB, (size_t) kLend * vram.slot_bytes()), "overwrite lent slots");
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
