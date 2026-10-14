// src/ds41/tests/vram_swap_test.cu - adaptive swaps with a RAM tier: after every switch between two steps, every
// expert is in VRAM or in host memory (its RAM slot or the swap buffer), never only in the file; when the swaps have
// landed, every VRAM slot and every RAM slot holds its expert's bytes. With the adaptive RAM tier, only experts in
// the RAM tier swap in.
// Runs on a fake pack (full 40 x 384 geometry, experts of 4, 8 and 12 KiB as SAGE's sizes differ).
#include "strata/ds41/host_experts.hpp"
#include "strata/ds41/vram_experts.hpp"
#include "bench_util.hpp"
#include "fake_pack.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using namespace ds41test;
namespace sd = strata::ds41;

namespace {

/// With the adaptive RAM tier, a swap takes `in` from its RAM slot only: an expert in neither tier (the tier did not
/// admit it, for example a skipped SSD expert) stays out of VRAM. Copying it from the mapped pack page-faulted it in
/// from the SSD during decode (laptop, 128K prompt: 1.5 GiB of experts.bin pages in 255 steps).
void adaptive_ram_tier(Verdict& v) {
    const std::string dir = "/tmp/ds41_fake_pack_swap_adapt", prof = "/tmp/ds41_fake_profile_swap_adapt.bin";
    write_fake_pack(dir);   // experts of one size: every swap fits
    std::vector<std::pair<int, int>> ranked;   // expert-major: VRAM holds experts 0..4 of every layer
    for (int e = 0; e < E; ++e)
        for (int l = 0; l < L; ++l) ranked.push_back({l, e});
    write_profile(prof, ranked);
    sd::Pack pack(dir);
    pack.map_experts();
    sd::VramExperts::Adapt ad;
    ad.every = 1;
    ad.max_swaps = 96;
    sd::VramExperts vram(pack, prof, 200, 0, ad);
    uint64_t total = 0;
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e) total += pack.expert(l, e).bytes;
    sd::HostExperts host(pack, ranked, vram.res_host(), total / 2, {}, 2);   // about experts 5..191 of every layer
    vram.set_host(&host);
    host.enable_adapt(4);
    std::vector<char> neither((size_t) L * E, 0);
    int n_neither = 0;
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e)
            if (vram.res_host()[(size_t) l * E + e] < 0 && !host.in_memory(l, e)) {
                neither[(size_t) l * E + e] = 1;
                ++n_neither;
            }
    v.check(n_neither > 0, "adaptive: some experts are in neither tier (" + std::to_string(n_neither) + ")");
    const std::vector<int32_t> start = vram.res_host();
    bool kept_out = true;
    for (int step = 0; step < 60; ++step) {
        std::vector<int32_t> routes((size_t) L * sd::kTopK);
        for (int l = 0; l < L; ++l)   // layer 0 and the last 4 only: one batch of swaps covers them
            for (int k = 0; k < sd::kTopK; ++k)
                routes[(size_t) l * sd::kTopK + k] = l == 0 || l >= L - 4 ? (step / 10 * 6 + k + l * 5) % E : -1;
        vram.count(routes.data(), sd::kTopK);
        vram.between_steps();
        for (size_t i = 0; i < neither.size(); ++i) kept_out &= !(neither[i] && vram.res_host()[i] >= 0);
    }
    vram.lend(0);
    vram.restore();
    int entered = 0;   // experts that swapped in
    for (size_t i = 0; i < neither.size(); ++i) {
        kept_out &= !(neither[i] && vram.res_host()[i] >= 0);
        entered += start[i] < 0 && vram.res_host()[i] >= 0;
    }
    // layer 0's routes use experts of the RAM tier, the last layers' ones (from expert 180 up) partly experts in neither
    v.check(entered > 0, "adaptive: experts from the RAM tier still swap in (" + std::to_string(entered) + ")");
    v.check(kept_out, "adaptive: no expert in neither tier enters VRAM");
    v.metric("adapt_neither", n_neither);
    v.metric("adapt_entered", entered);
}

}  // namespace

int main() {
    Verdict v;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("RESULT skip (no GPU)\n");
        return 77;
    }
    const std::string dir = "/tmp/ds41_fake_pack_swap", prof = "/tmp/ds41_fake_profile_swap.bin";
    write_fake_pack(dir, true);
    std::vector<std::pair<int, int>> ranked;
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e) ranked.push_back({l, (e * 7 + l) % E});
    write_profile(prof, ranked);
    sd::Pack pack(dir);
    pack.map_experts();
    sd::VramExperts::Adapt ad;
    ad.every = 1;
    ad.max_swaps = 24;
    sd::VramExperts vram(pack, prof, 200, 0, ad);   // the first 200 ranked, all in layer 0
    sd::HostExperts host(pack, ranked, vram.res_host(), 1ull << 40, {}, 2);   // every other expert in RAM
    vram.set_host(&host);
    v.check(host.slots() == L * E - vram.slots(), "the RAM tier holds every expert outside VRAM");

    auto everywhere = [&](const char* when) {
        int file_only = 0;
        for (int l = 0; l < L; ++l)
            for (int e = 0; e < E; ++e)
                if (vram.res_host()[(size_t) l * E + e] < 0 && !host.in_memory(l, e)) ++file_only;
        if (file_only) std::printf("  %s: %d experts only in the file\n", when, file_only);
        return file_only == 0;
    };
    // routing that favours experts outside VRAM (layer 0's last ones, every other layer's first ones): they replace
    // the least used residents, batch after batch
    bool always = everywhere("start");
    int64_t swaps = 0;
    for (int step = 0; step < 60; ++step) {
        std::vector<int32_t> routes((size_t) L * sd::kTopK);
        for (int l = 0; l < L; ++l)
            for (int k = 0; k < sd::kTopK; ++k) routes[(size_t) l * sd::kTopK + k] = (step / 10 * 6 + k + l * 5) % E;
        vram.count(routes.data(), sd::kTopK);
        swaps += vram.between_steps();
        always &= everywhere(("step " + std::to_string(step)).c_str());
    }
    v.check(swaps > 0, "the routing moved experts (" + std::to_string(swaps) + " swaps)");
    v.check(always, "after every switch, every expert is in VRAM or in host memory");
    // flush the batch in flight (lend commits it), then compare every byte
    vram.lend(0);
    vram.restore();
    v.check(everywhere("flushed"), "after the last batch, too");
    bool vram_ok = true, ram_ok = true;
    std::vector<uint8_t> buf(3 * kExpertBytes);
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e) {
            const sd::ExpertSlot& x = pack.expert(l, e);
            const uint8_t* want = pack.expert_base() + x.offset;
            const int32_t s = vram.res_host()[(size_t) l * E + e];
            if (s >= 0) {
                const uint8_t* dev = (const uint8_t*) vram.desc(s).w1.trellis - x.comp_off[0];
                vram_ok &= cudaMemcpy(buf.data(), dev, x.bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                           std::memcmp(buf.data(), want, x.bytes) == 0;
            }
            const int32_t r = host.slot_of(l, e);
            if (r >= 0) ram_ok &= std::memcmp(host.slot_ptr(r), want, x.bytes) == 0;
            if (s >= 0 && r >= 0) ram_ok = false;   // one tier each
        }
    v.check(vram_ok, "every VRAM slot holds its expert's bytes");
    v.check(ram_ok, "every RAM slot holds its expert's bytes, and no expert is in both tiers");
    adaptive_ram_tier(v);
    return v.finish();
}
