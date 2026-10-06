// Test tier metadata with a mixed-size fake pack. Never run K10 on these tiny experts.
#include "bench_util.hpp"
#include "fake_pack.hpp"
#include "strata/ds41/host_experts.hpp"
#include "strata/ds41/vram_experts.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <unistd.h>

using namespace ds41test;
namespace sd = strata::ds41;
namespace kk = strata::ds41::kernels;

__global__ void read_word(const kk::Exl3Expert* table, int index, uint16_t* out) {
    *out = *table[index].w1.trellis;
}

int main() {
    require_gpu();
    Verdict v;
    char path[] = "/tmp/ds41-hybrid-XXXXXX";
    if (!mkdtemp(path)) return 1;
    write_fake_pack(path, true);
    {
        sd::Pack pack(path);
        pack.map_experts();
        std::vector<int32_t> res(size_t(L) * E, -1);
        const std::vector<std::pair<int, int>> rank{{0, 2}, {1, 0}};
        sd::HostExperts empty(pack, rank, res, 0, {}, 1);
        v.check(empty.experts_dev() == nullptr, "zero budget has no mapped table");
        sd::HostExperts host(pack, rank, res, 4 * kExpertBytes, {}, 2);
        v.check(host.experts_dev() != nullptr, "RAM mapping must work on the GPU test host");
        if (!host.experts_dev()) return v.finish();
        const auto* stable = host.experts_dev();
        auto read = [&](int l, int e) {
            kk::Exl3Expert d{};
            ck(cudaMemcpy(&d, stable + size_t(l) * E + e, sizeof(d), cudaMemcpyDeviceToHost), "table read");
            return d;
        };
        Dev<uint16_t> word(1);
        cudaStream_t stream;
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "table reader stream");
        auto check_entry = [&](int l, int e) {
            const int slot = host.slot_of(l, e);
            const auto d = read(l, e);
            if (slot < 0) { v.check(d.w1.trellis == nullptr, "file descriptor is absent"); return; }
            void* alias = nullptr;
            ck(cudaHostGetDevicePointer(&alias, host.slot_ptr(0), 0), "arena alias");
            const auto* base = (const uint8_t*) alias + (host.slot_ptr(slot) - host.slot_ptr(0));
            const auto expected = sd::VramExperts::describe_at(pack, l, e, base);
            auto same = [](const kk::Exl3Proj& a, const kk::Exl3Proj& b) {
                return a.trellis == b.trellis && a.suh == b.suh && a.svh == b.svh &&
                       a.k == b.k && a.n == b.n && a.tile_w == b.tile_w;
            };
            v.check(same(d.w1, expected.w1) && same(d.w3, expected.w3) && same(d.w2, expected.w2),
                    "descriptor uses the current expert offsets and device alias");
            // Prove the published pointer reaches the current bytes, even when the UVA addresses match.
            uint16_t want = 0;
            read_word<<<1, 1, 0, stream>>>(stable, l * E + e, word.p);
            ck(cudaStreamSynchronize(stream), "mapped read on nonblocking stream");
            std::memcpy(&want, host.slot_ptr(slot) + pack.expert(l, e).comp_off[0], 2);
            v.check(word.down()[0] == want, "device descriptor reaches RAM bytes");
        };
        check_entry(0, 2);
        check_entry(1, 0);
        check_entry(2, 0);
        const int slot = host.slot_of(0, 2);
        for (int e : {3, 4, 5, 6}) {
            const int old = e == 3 ? 2 : e - 1;
            host.point_to_file(0, old);
            check_entry(0, old);
            const auto& s = pack.expert(0, e);
            std::memcpy(host.slot_ptr(slot), pack.expert_base() + s.offset, s.bytes);
            host.assign(slot, 0, e);
            check_entry(0, old);
            check_entry(0, e);
            v.check(host.experts_dev() == stable, "table address stays stable across swaps");
        }
        bool threw = false;
        try { host.assign(host.slot_of(1, 0), 2, 2); }
        catch (const std::invalid_argument&) { threw = true; }
        v.check(threw, "oversize assignment rejected");
        check_entry(1, 0);
        check_entry(2, 2);
        cudaStreamDestroy(stream);
    }
    {
        // Exercise the real VRAM swap callbacks. Both experts fit the same RAM slot.
        sd::Pack pack(path);
        pack.map_experts();
        const std::vector<std::pair<int, int>> rank{{0, 2}, {0, 5}};
        const std::string profile = std::string(path) + "/profile.bin";
        write_profile(profile, rank);
        sd::VramExperts::Adapt adapt;
        adapt.every = 1;
        adapt.max_swaps = 1;
        // Destroy VRAM first. Its copy thread may access the RAM tier.
        std::unique_ptr<sd::HostExperts> host;
        sd::VramExperts vram(pack, profile, 1, 0, adapt);
        host = std::make_unique<sd::HostExperts>(pack, rank, vram.res_host(), 3 * kExpertBytes,
                                                std::vector<int64_t>{}, 1);
        v.check(host->experts_dev() != nullptr, "swap tier is mapped");
        if (host->experts_dev()) {
            vram.set_host(host.get());
            const auto* table = host->experts_dev();
            auto entry = [&](int e) {
                kk::Exl3Expert d{};
                ck(cudaMemcpy(&d, table + e, sizeof(d), cudaMemcpyDeviceToHost), "swap entry");
                return d;
            };
            v.check(!entry(2).w1.trellis && entry(5).w1.trellis, "initial tier complement");
            std::vector<int32_t> routes(size_t(L) * 6, -1);
            routes[0] = 5;
            for (int i = 0; i < 3; ++i) vram.count(routes.data(), 6);
            vram.between_steps();
            v.check(!entry(2).w1.trellis && !entry(5).w1.trellis, "in-flight swap revokes RAM entry");
            v.check(vram.res_host()[2] < 0 && vram.res_host()[5] < 0, "in-flight swap revokes VRAM entry");
            vram.lend(0); // Join the copier and commit at this safe point.
            v.check(entry(2).w1.trellis && !entry(5).w1.trellis, "commit publishes new RAM holder");
            v.check(vram.res_host()[5] == 0 && host->slot_of(0, 2) == 0, "swap updates both tiers");
            const auto& old = pack.expert(0, 2);
            v.check(std::memcmp(host->slot_ptr(0), pack.expert_base() + old.offset, old.bytes) == 0,
                    "evicted VRAM bytes reach the RAM slot");
            v.check(host->experts_dev() == table, "swap keeps the table address");
        }
    }
    std::filesystem::remove_all(path);
    return v.finish();
}
