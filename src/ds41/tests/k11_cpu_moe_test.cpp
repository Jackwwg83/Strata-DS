// src/ds41/tests/k11_cpu_moe_test.cpp - task K11 acceptance: the CPU EXL3 expert kernel (exllamav3 moe_mul1, vendored in
// third_party/exllamav3_moe) on the K10 golden inputs: 32 real experts of layer 10, m = 1, 4, 8. Fixed by
// ds41/tasks/K11.md.
//   accuracy: relative L2 against the FP16 golden (exllamav3 LinearEXL3, EXL3_INT8_GEMV=0) at most kMaxErr. The
//             vendored kernel quantizes activations to int8, so it is not exact; kMaxErr is its measured error + 10%.
//   speed:    median time of one forward, m = 1 and m = 8, with K11_THREADS threads (default 8: the P-cores of the
//             queue box's i9-14900K). The experts are copied to RAM first: the kernel, not the SSD, is measured.
// Env: K11_PACK (default /workspace/pack-3bpw), K11_GOLDEN (default /workspace/ci/golden/k10), K11_THREADS.
#include "moe_mul1.h"
#include "strata/ds41/pack.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace sd = strata::ds41;

namespace {

constexpr int H = 5120, F = 2304, K = 6;
constexpr double kMaxErr = 0.0325;   // the vendored kernel measured 0.0279 / 0.0289 / 0.0295 (m = 1, 4, 8) + 10%

int failures = 0;
std::string metrics;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what.c_str()); }
}
void metric(const char* k, double v) {
    char b[96];
    std::snprintf(b, sizeof b, " %s=%.4g", k, v);
    metrics += b;
}

template <typename T>
std::vector<T> read_bin(const std::string& path, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (n * sizeof(T)))) {
        std::printf("RESULT fail missing-golden=%s\n", path.c_str());
        std::exit(1);
    }
    return v;
}

at::Half to_half(float f) {   // round to nearest even, normal range (routing weights)
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const uint32_t s = (b >> 16) & 0x8000u;
    const int e = (int) ((b >> 23) & 0xff) - 127 + 15;
    if (e <= 0) return at::Half((uint16_t) s, at::Half::from_bits());
    uint32_t m = b & 0x7fffff;
    uint32_t h = s | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (h & 1))) ++h;
    return at::Half((uint16_t) h, at::Half::from_bits());
}

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main() {
    const char* pe = std::getenv("K11_PACK");
    const char* ge = std::getenv("K11_GOLDEN");
    const char* te = std::getenv("K11_THREADS");
    const std::string pack_dir = pe ? pe : "/workspace/pack-3bpw", gold = ge ? ge : "/workspace/ci/golden/k10";
    const int threads = te ? std::atoi(te) : 8;
    sd::Pack pack(pack_dir);
    pack.map_experts();
    std::vector<std::pair<int, int>> le;
    {
        std::ifstream f(gold + "/experts.txt");
        int l, e;
        while (f >> l >> e) le.push_back({l, e});
    }
    if (le.empty()) { std::printf("RESULT fail missing-golden=%s/experts.txt\n", gold.c_str()); return 1; }
    // copy the experts to RAM, register them as one layer of le.size() experts
    std::vector<std::vector<uint8_t>> ram(le.size());
    std::vector<MoeCpuMatrixDesc> g(le.size()), u(le.size()), d(le.size());
    for (size_t i = 0; i < le.size(); ++i) {
        const auto& s = pack.expert(le[i].first, le[i].second);
        ram[i].assign(pack.expert_base() + s.offset, pack.expert_base() + s.offset + s.bytes);
        const uint8_t* b = ram[i].data();
        auto desc = [&](int c0, int k_tiles, int n_tiles) {
            MoeCpuMatrixDesc m;
            m.trellis = (const uint16_t*) (b + s.comp_off[c0]);
            m.suh = (const at::Half*) (b + s.comp_off[c0 + 1]);
            m.svh = (const at::Half*) (b + s.comp_off[c0 + 2]);
            m.k_tiles = k_tiles;
            m.n_tiles = n_tiles;
            m.tile_w = (int) (s.comp_bytes[c0] / ((uint64_t) k_tiles * n_tiles * 2));
            return m;
        };
        g[i] = desc(0, H / 16, F / 16);
        u[i] = desc(4, H / 16, F / 16);
        d[i] = desc(8, F / 16, H / 16);
    }
    const int64_t layer = exl3_moe_cpu_make_layer_raw(g.data(), u.data(), d.data(), (int) le.size(), 0, 10.0f, 0);
    double t1 = 0, t8 = 0;
    for (int m : {1, 4, 8}) {
        const std::string s = std::to_string(m);
        const auto x = read_bin<uint16_t>(gold + "/x_" + s + ".bin", (size_t) m * H);
        const auto sel = read_bin<int32_t>(gold + "/sel_" + s + ".bin", (size_t) m * K);
        const auto wf = read_bin<float>(gold + "/w_" + s + ".bin", (size_t) m * K);
        const auto want = read_bin<float>(gold + "/out_" + s + ".bin", (size_t) m * H);
        std::vector<at::Half> w(wf.size());
        for (size_t i = 0; i < wf.size(); ++i) w[i] = to_half(wf[i]);
        std::vector<float> out((size_t) m * H);
        auto run = [&] {
            exl3_moe_cpu_forward_raw(layer, (const at::Half*) x.data(), sel.data(), w.data(), out.data(), m, K, threads);
        };
        run();
        double num = 0, den = 0;
        bool finite = true;
        for (size_t i = 0; i < out.size(); ++i) {
            finite = finite && std::isfinite(out[i]);
            num += ((double) out[i] - want[i]) * ((double) out[i] - want[i]);
            den += (double) want[i] * want[i];
        }
        const double err = std::sqrt(num / std::max(den, 1e-300));
        std::printf("m=%d rel_l2 vs FP16 golden = %.5f\n", m, err);
        check(finite, "non-finite output");
        if (kMaxErr > 0) check(err <= kMaxErr, "relative L2 against the FP16 golden above the limit");
        metric(("err_m" + s).c_str(), err);
        if (m != 4) {
            std::vector<double> t;
            for (int r = 0; r < 3; ++r) run();
            for (int r = 0; r < 21; ++r) {
                const double a = now_us();
                run();
                t.push_back(now_us() - a);
            }
            std::sort(t.begin(), t.end());
            std::printf("  time m=%d, %d threads: %.0f us (%.0f us per expert slot)\n", m, threads, t[10],
                        t[10] / (m * K));
            (m == 1 ? t1 : t8) = t[10];
        }
    }
    exl3_moe_cpu_free_layer(layer);
    metric("us_m1", t1);
    metric("us_m8", t8);
    metric("score_us", t1 + t8 / 8);
    std::printf("RESULT %s%s\n", failures ? "fail" : "pass", metrics.c_str());
    return failures ? 1 : 0;
}
