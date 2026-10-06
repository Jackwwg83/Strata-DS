// src/ds41/tests/vram_lend_test.cu - VramExperts::lend / restore (prefill borrows cache slots): the lent slots leave
// the residency table, the caller may overwrite them, and restore() brings back every byte and table entry.
// Runs on a fake pack: full 40 x 384 geometry, 4 KiB experts of random bytes, no dense weights.
#include "strata/ds41/vram_experts.hpp"

#include "bench_util.hpp"
#include "strata/ds41/config.hpp"

#include <sys/stat.h>

#include <cstdio>
#include <fstream>
#include <string>

using namespace ds41test;
namespace sd = strata::ds41;

namespace {

constexpr int L = sd::kLayers, E = sd::kExperts;
constexpr uint64_t kExpertBytes = 4096;

void write_fake_pack(const std::string& dir) {
    mkdir(dir.c_str(), 0755);
    {
        std::ofstream f(dir + "/index.txt");
        f << "embed.weight bf16 2 " << sd::kVocab << " " << sd::kDim << " 0 " << (uint64_t) sd::kVocab * sd::kDim * 2 << "\n";
        f << "head.weight bf16 2 " << sd::kVocab << " " << sd::kDim << " 0 " << (uint64_t) sd::kVocab * sd::kDim * 2 << "\n";
        f << "layers.0.attn.wq_b.weight f8e4m3 2 " << sd::kHeads * sd::kHeadDim << " " << sd::kQLora << " 0 "
          << (uint64_t) sd::kHeads * sd::kHeadDim * sd::kQLora << "\n";
        f << "layers.0.attn.wo_a.weight bf16 2 " << sd::kOGroups * sd::kOLora << " " << sd::kHeads * sd::kHeadDim / sd::kOGroups
          << " 0 " << (uint64_t) sd::kOGroups * sd::kOLora * (sd::kHeads * sd::kHeadDim / sd::kOGroups) * 2 << "\n";
        f << "layers.0.hc_attn_fn f32 2 " << sd::kHcMix << " " << sd::kHc * sd::kDim << " 0 "
          << (uint64_t) sd::kHcMix * sd::kHc * sd::kDim * 4 << "\n";
    }
    {
        std::ofstream f(dir + "/experts.txt");
        const char* comp[12] = {"w1.trellis", "w1.suh", "w1.svh", "w1.mul1", "w3.trellis", "w3.suh",
                                "w3.svh",     "w3.mul1", "w2.trellis", "w2.suh", "w2.svh", "w2.mul1"};
        for (int l = 0; l < L; ++l)
            for (int e = 0; e < E; ++e) {
                f << l << " " << e << " " << ((uint64_t) l * E + e) * kExpertBytes << " " << kExpertBytes << " 3 3 3";
                for (int c = 0; c < 12; ++c) f << " " << comp[c] << ":" << c * 256 << ":256";
                f << "\n";
            }
    }
    {
        std::vector<uint8_t> b((size_t) L * E * kExpertBytes);
        std::mt19937 g(1);
        for (auto& x : b) x = (uint8_t) g();
        std::ofstream f(dir + "/experts.bin", std::ios::binary);
        f.write((const char*) b.data(), (std::streamsize) b.size());
    }
    std::ofstream(dir + "/engram.txt");
    std::ofstream(dir + "/engram_hash.txt");
    {
        std::vector<int32_t> tm(sd::kVocab, 0);
        std::ofstream f(dir + "/engram_tokenmap.bin", std::ios::binary);
        f.write((const char*) tm.data(), (std::streamsize) (tm.size() * 4));
    }
    std::ofstream(dir + "/pack_info.txt") << "layers " << L << "\nexperts " << E << "\nfinished 1\n";
}

void write_profile(const std::string& path, const std::vector<std::pair<int, int>>& pairs) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    const uint32_t n = (uint32_t) pairs.size();
    const uint32_t h[5] = {1, (uint32_t) L, (uint32_t) E, n, n};
    std::fwrite("STRP", 1, 4, f);
    std::fwrite(h, 4, 5, f);
    for (auto [l, e] : pairs) {
        const uint16_t p[2] = {(uint16_t) l, (uint16_t) e};
        std::fwrite(p, 2, 2, f);
    }
    std::vector<int32_t> table((size_t) L * E, -1);
    for (uint32_t i = 0; i < n; ++i) table[(size_t) pairs[i].first * E + pairs[i].second] = (int32_t) i;
    std::fwrite(table.data(), 4, table.size(), f);
    std::fclose(f);
}

}  // namespace

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
