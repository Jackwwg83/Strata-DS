// Synthetic packs and host fault injection. No GPU results are claimed.
#include "strata/ds41/pack.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/engram_rows.hpp"
#include "strata/ds41/expert_stream.hpp"
#include "strata/ds41/host_experts.hpp"
#include "strata/ds41/vram_experts.hpp"
#include "strata/ds41/lookahead.hpp"
#include "moe_mul1.h"
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
using namespace strata::ds41;
namespace fs = std::filesystem;
extern thread_local int revfix_allocation_fail;
void exl3_moe_cpu_set_expert_raw(int64_t, int, const MoeCpuMatrixDesc*, const MoeCpuMatrixDesc*,
                               const MoeCpuMatrixDesc*, int) {}
void require(bool b, const char* msg) { if (!b) throw std::runtime_error(msg); }
template<class F> void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    require(threw, "invalid input was accepted");
}
void put(const fs::path& d, const char* n, const std::string& s) { std::ofstream(d/n) << s; }
void fixture(const fs::path& d, const std::string& mode) {
    put(d, "pack_info.txt", "finished 1\nlayers 40\nexperts 384\n");
    std::string index = "embed.weight u8 2 129280 5120 0 661913600\nhead.weight u8 2 129280 5120 0 661913600\n"
        "layers.0.attn.wq_b.weight u8 2 32768 1280 0 41943040\n"
        "layers.0.attn.wo_a.weight u8 2 8192 4096 0 33554432\n"
        "layers.0.hc_attn_fn u8 2 24 20480 0 491520\n";
    if (mode == "dense_overflow") index += "z u8 1 512 18446744073709551360 512\n";
    if (mode == "shape_overflow") index += "z u8 3 4294967296 4294967296 2 0 0\n";
    put(d, "index.txt", index);
    const char* names[] = {"w1.trellis", "w1.suh", "w1.svh", "w1.mul1", "w3.trellis", "w3.suh",
        "w3.svh", "w3.mul1", "w2.trellis", "w2.suh", "w2.svh", "w2.mul1"};
    std::ofstream ex(d/"experts.txt");
    for (int l = 0; l < kLayers; ++l) for (int e = 0; e < kExperts; ++e) {
        const bool first = l == 0 && e == 0;
        std::string off = first && mode == "early_slot" ? "8192" : "0";
        if (mode == "slot_overflow" && l == kLayers-1 && e == kExperts-1) off = "18446744073709551360";
        ex << l << ' ' << e << ' ' << off << " 4096 3 3 3";
        for (int c = 0; c < 12; ++c)
            ex << ' ' << names[c] << (first && c == 0 && mode == "component_overflow" ?
                ":18446744073709551360:512" : ":0:2");
        ex << '\n';
    }
    put(d, "experts.bin", std::string(4096, 'x'));
    fs::create_directories(d / "DeepSeek Flash");
    put(d / "DeepSeek Flash", "table.safetensors", std::string(32768, 'r'));
    put(d, "engram.txt", "1 100 256 0 25600 " + (d / "DeepSeek Flash/table.safetensors").string() + "\n");
    std::string hash = "max_ngram 2\nn_heads 1\npad 0\nvocab 10\nlayers 1\n";
    if (mode != "missing_index") hash += "multipliers " + std::string(mode == "negative_index" ? "-1" : "0") + " 1 2\n";
    hash += mode == "zero_prime" ? "primes 0 0\n" : "primes 0 7\n";
    hash += "offsets 0 0\n";
    if (mode == "bad_index") hash += "multipliers nope 1 2\n";
    if (mode == "large_index") hash += "multipliers 99999 1 2\n";
    if (mode == "short_array") hash += "primes 0\n";
    if (mode == "layer_mismatch") hash += "layers 14\n";
    put(d, "engram_hash.txt", hash);
    put(d, "engram_tokenmap.bin", std::string(kVocab * 4, '\0'));
    std::ofstream profile(d/"profile", std::ios::binary);
    uint32_t header[] = {1, kLayers, kExperts, 0, 1}; uint16_t pair[] = {0,0};
    profile.write("STRP",4); profile.write((char*)header,sizeof header); profile.write((char*)pair,sizeof pair);
}
int fd_count() { int n = 0; for (int i = 0; i < 1024; ++i) if (fcntl(i,F_GETFD) >= 0) ++n; return n; }
// Hold the callback open. A residency write must wait for its read section.
template<class F> void blocks_writer(F write, std::function<void()> read = [] {}) {
    std::promise<void> entered, release, started;
    auto gate = release.get_future().share();
    RouterLookahead look({{0},{0}}, {{0},{0}}, 1, 1, 1, [&](int,int) {
        read(); entered.set_value(); gate.wait(); read(); return false;
    });
    uint16_t x = 0; look.post(0,&x); entered.get_future().wait();
    auto writer = std::async(std::launch::async, [&] { started.set_value(); write(); });
    started.get_future().wait();
    const bool blocked = writer.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout;
    release.set_value(); writer.get();
    require(blocked, "residency writer ran during the lookahead callback");
}
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const std::string mode = argv[1]; const fs::path d = argv[2];
    fs::create_directories(d); fixture(d,mode);
    try {
        if (mode == "early_slot" || mode == "slot_overflow") {
            Pack p(d); rejects([&] { p.map_experts(); });
            require(!p.expert_base(), "invalid mapping was published");
        } else if (mode == "dense_overflow") {
            fs::resize_file(d/"experts.bin",4096);
            put(d,"dense.bin",""); fs::resize_file(d/"dense.bin",661913600);
            Pack p(d); rejects([&] { p.upload_dense(); });
        } else if (mode == "path_spaces") {
            Pack p(d); const auto& t = p.engram_tables()[0];
            require(t.path == (d / "DeepSeek Flash/table.safetensors").string(), "path truncated");
            EngramRows rows({{t.path, t.weight_offset, t.scale_offset}}, 1, 256, 8, 1);
            int64_t id = 0; uint8_t w[256]{}, scale[8]{};
            rows.read({&id}, 1, {w}, {scale});
            require(w[255] == 'r' && scale[7] == 'r', "parsed path did not reach the row reader");
        } else if (mode == "vram_cleanup" || mode == "stream_cleanup") {
            Pack p(d); p.map_experts(); const int fds = fd_count();
            if (mode == "vram_cleanup") {
                rejects([&] { VramExperts v(p,(d/"missing").string(),1,0,{}); });
                require(fake_cuda::live == 0, "CUDA resources leaked after missing profile");
            }
            const int count = mode == "vram_cleanup" ? 5 : 11;
            for (int at = 0; at < count; ++at) {
                fake_cuda::calls = 0; fake_cuda::fail_at = at;
                rejects([&] {
                    if (mode == "vram_cleanup") { VramExperts v(p,(d/"profile").string(),1,0,{}); }
                    else { ExpertStream s(p,nullptr,nullptr,2,4096,2,3,false); }
                });
                fake_cuda::fail_at = -1;
                require(fake_cuda::live == 0, "CUDA resources leaked after injected failure");
                require(fd_count() == fds, "file descriptor leaked");
            }
            if (mode == "stream_cleanup") { ExpertStream s(p,nullptr,nullptr,2,4096,2,3,false); }
            require(fake_cuda::live == 0, "normal destruction leaked");
        } else if (mode == "thread_cleanup") {
            Pack p(d); p.map_experts(); const int fds = fd_count();
            bool success = false;
            for (int at = 0; at < 40; ++at) {
                revfix_allocation_fail = at;
                try { ExpertStream stream(p,nullptr,nullptr,2,4096,2,3,false); success = true; }
                catch (const std::exception&) {}
                revfix_allocation_fail = -1;
                std::cout << "allocation point " << at << (success ? " completed" : " rejected") << std::endl;
                require(fake_cuda::live == 0, "allocation failure leaked CUDA resources");
                require(fd_count() == fds, "allocation failure leaked a file descriptor");
                if (success) break;
            }
            require(success, "allocation sweep did not reach normal construction");
        } else if (mode == "vram_race" || mode == "host_race") {
            Pack p(d); p.map_experts();
            VramExperts v(p,(d/"profile").string(),1,0,{});
            if (mode == "vram_race") {
                auto read = [&] { volatile int32_t entry = v.res_host()[0]; (void) entry; };
                blocks_writer([&] { v.lend(1); }, read);
                blocks_writer([&] { v.restore(); }, read);
            } else {
                HostExperts h(p,{{0,1}},v.res_host(),4096,{},1);
                auto read = [&] { volatile int32_t entry = h.slot_of(0,1); (void) entry; };
                blocks_writer([&] { h.point_to_file(0,1); }, read);
                blocks_writer([&] { h.assign(0,0,2); }, read);
            }
        } else if (mode == "residency_stress") {
            Pack p(d); p.map_experts();
            VramExperts v(p,(d/"profile").string(),1,0,{});
            HostExperts h(p,{{0,1}},v.res_host(),4096,{},1);
            std::atomic<int> reads{0};
            RouterLookahead look({{0},{0}}, {{0},{0}}, 1, 1, 1, [&](int,int) {
                for (int i = 0; i < 100; ++i) {
                    volatile int32_t vr = v.res_host()[0], ram = h.slot_of(0,1);
                    (void) vr; (void) ram;
                    reads.fetch_add(1, std::memory_order_relaxed);
                }
                return false;
            });
            uint16_t x = 0;
            look.post(0,&x);
            while (!reads.load()) std::this_thread::yield();
            for (int i = 0; i < 1000; ++i) {
                look.post(0,&x);
                v.lend(1); h.point_to_file(0,1); h.assign(0,0,1); v.restore();
            }
            require(reads > 0, "lookahead never read residency");
        } else if (mode == "engram_retry") {
            put(d,"rows",std::string(16384,'a'));
            EngramRows er({{(d/"rows").string(),0,8192},{(d/"rows").string(),0,8192}},2,256,8,2);
            int64_t bad[] = {999999,0}, good[] = {0,1}; uint8_t w[1024]{}, s[32]{};
            rejects([&] { er.read({bad,good},2,{w,w+512},{s,s+16}); });
            bool poisoned = false;
            try { er.read({good,good},1,{w,w+512},{s,s+16}); }
            catch (const std::exception& e) { poisoned = std::string(e.what()).find("failed state") != std::string::npos; }
            require(poisoned, "failed reader accepted reuse or consumed stale completions");
        } else if (mode == "valid") { Pack p(d); p.map_experts(); }
        else { rejects([&] { Pack p(d); }); }
        std::cout << "PASS " << mode << '\n'; return 0;
    } catch (const std::exception& e) { std::cout << "FAIL " << mode << ": " << e.what() << '\n'; return 1; }
}
