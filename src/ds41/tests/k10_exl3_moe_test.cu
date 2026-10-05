// src/ds41/tests/k10_exl3_moe_test.cu - task K10 acceptance: golden outputs from exllamav3 LinearEXL3 on 32 real
// experts of the ds41 pack (ds41/ci/make_k10_golden.py), then speed. Fixed by ds41/tasks/K10.md.
// Env: K10_PACK (default /workspace/pack-3bpw), K10_GOLDEN (default /workspace/ci/golden/k10).
#include "bench_util.hpp"

#include "strata/ds41/kernels/k10_exl3_moe.hpp"
#include "strata/ds41/pack.hpp"

#include <fstream>

using namespace ds41test;
namespace sd = strata::ds41;
namespace kk = strata::ds41::kernels;

template <typename T>
static std::vector<T> read_bin(const std::string& path, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (n * sizeof(T)))) {
        std::printf("RESULT fail missing-golden=%s\n", path.c_str());
        std::exit(1);
    }
    return v;
}

int main() {
    require_gpu();
    const char* pe = std::getenv("K10_PACK");
    const char* ge = std::getenv("K10_GOLDEN");
    const std::string pack_dir = pe ? pe : "/workspace/pack-3bpw", gold = ge ? ge : "/workspace/ci/golden/k10";
    Verdict v;
    sd::Pack pack(pack_dir);
    pack.map_experts();
    // the 32 golden experts, each copied into its own device slot as the pack lays it out
    std::vector<std::pair<int, int>> le;
    {
        std::ifstream f(gold + "/experts.txt");
        int l, e;
        while (f >> l >> e) le.push_back({l, e});
    }
    std::vector<kk::Exl3Expert> host(le.size());
    std::vector<void*> slots;
    for (size_t i = 0; i < le.size(); ++i) {
        const auto& s = pack.expert(le[i].first, le[i].second);
        void* d = nullptr;
        ck(cudaMalloc(&d, s.bytes), "slot");
        ck(cudaMemcpy(d, pack.expert_base() + s.offset, s.bytes, cudaMemcpyHostToDevice), "slot copy");
        slots.push_back(d);
        auto proj = [&](int c0, int k, int n) {
            kk::Exl3Proj p;
            p.trellis = (const uint16_t*) ((char*) d + s.comp_off[c0]);
            p.suh = (const __half*) ((char*) d + s.comp_off[c0 + 1]);
            p.svh = (const __half*) ((char*) d + s.comp_off[c0 + 2]);
            p.k = k;
            p.n = n;
            p.tile_w = (int) (s.comp_bytes[c0] / ((uint64_t) (k / 16) * (n / 16) * 2));
            return p;
        };
        host[i].w1 = proj(0, 5120, 2304);
        host[i].w3 = proj(4, 5120, 2304);
        host[i].w2 = proj(8, 2304, 5120);
    }
    Dev<kk::Exl3Expert> experts(host);
    const size_t ws_bytes = 64ull << 20;
    Dev<uint8_t> ws(ws_bytes);
    double us1 = 0, us8 = 0;
    for (int m : {1, 4, 8}) {
        const std::string s = std::to_string(m);
        Dev<__half> x(read_bin<__half>(gold + "/x_" + s + ".bin", (size_t) m * 5120));
        Dev<int32_t> sel(read_bin<int32_t>(gold + "/sel_" + s + ".bin", (size_t) m * 6));
        Dev<float> w(read_bin<float>(gold + "/w_" + s + ".bin", (size_t) m * 6));
        const auto want = read_bin<float>(gold + "/out_" + s + ".bin", (size_t) m * 5120);
        Dev<float> out(std::vector<float>((size_t) m * 5120, 0.0f));
        kk::exl3_moe_decode(x.p, m, sel.p, w.p, 6, experts.p, out.p, ws.p, ws_bytes, 0);
        ck(cudaDeviceSynchronize(), "run");
        const double err = rel_l2(out.down(), want);
        std::printf("m=%d rel_l2=%.3g\n", m, err);
        v.check(err <= 5e-3, "relative L2 error above 5e-3 against exllamav3");
        if (m != 4) {
            const double us = median_us([&] { kk::exl3_moe_decode(x.p, m, sel.p, w.p, 6, experts.p, out.p, ws.p, ws_bytes, 0); });
            std::printf("  time m=%d: %.1f us\n", m, us);
            (m == 1 ? us1 : us8) = us;
        }
    }
    for (void* p : slots) cudaFree(p);
    v.metric("us_m1", us1);
    v.metric("us_m8", us8);
    v.metric("score_us", us1 + us8 / 8);
    return v.finish();
}
