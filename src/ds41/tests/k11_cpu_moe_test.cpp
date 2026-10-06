// src/ds41/tests/k11_cpu_moe_test.cpp - task K11 acceptance: the CPU EXL3 expert kernel (exllamav3 moe_mul1, vendored in
// third_party/exllamav3_moe) on the K10 golden inputs: 32 real experts of layer 10, m = 1, 4, 8. Fixed by
// ds41/tasks/K11.md.
//   accuracy: relative L2 against the FP16 golden (exllamav3 LinearEXL3, EXL3_INT8_GEMV=0) at most kMaxErr. The
//             vendored kernel quantizes activations to int8, so it is not exact; kMaxErr is its measured error + 10%.
//   speed:    median time of one forward, m = 1 and m = 8, with K11_THREADS threads (default 8: the P-cores of the
//             queue box's i9-14900K). The experts are copied to RAM first: the kernel, not the SSD, is measured.
// Env: K11_PACK (default /workspace/pack-3bpw), K11_GOLDEN (default /workspace/ci/golden/k10), K11_THREADS,
//      K11_WEIGHTS: ram (default, the only scored mode) | mmap-warm | mmap-ptecold | mmap-cold | all. The mmap modes
//      read the experts through the test's own shared mapping of experts.bin, as the engine's file tier does:
//      mmap-warm: every page cached (checked with mincore before each timed forward; all pages, no tolerance);
//      mmap-ptecold: a fresh mapping at the same address (MAP_FIXED) before each timed forward, every page cached;
//      mmap-cold: a fresh mapping and POSIX_FADV_DONTNEED before each timed forward, no page cached (mincore).
//      A mode whose state cannot be established is reported INVALID and not timed. Every forward is checked.
//      They print metrics (and the storage MB read per forward, /proc/self/io), never the score.
#include "moe_mul1.h"
#include "mixedk_fixture.hpp"
#include "strata/ds41/pack.hpp"

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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

/// bytes this process read from storage so far (/proc/self/io read_bytes), in MB; -1 when unavailable
double read_bytes_mb() {
    std::ifstream f("/proc/self/io");
    std::string k;
    unsigned long long v = 0;
    while (f >> k >> v)
        if (k == "read_bytes:") return v / 1e6;
    return -1;
}

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The same FP16 LinearEXL3 reference and tolerance as the fixed cases.
void mixed_k(int threads) {
    auto data = mixedk::load();
    auto relocated = data;
    auto desc = [](const mixedk::Projection& p) {
        return MoeCpuMatrixDesc{p.trellis.data(), reinterpret_cast<const at::Half*>(p.suh.data()),
                                reinterpret_cast<const at::Half*>(p.svh.data()), p.k / 16, p.n / 16, 16 * p.bits};
    };
    std::vector<MoeCpuMatrixDesc> gate, up, down;
    for (int e = 0; e < mixedk::E; ++e) {
        gate.push_back(desc(data[3 * e]));
        up.push_back(desc(data[3 * e + 1]));
        down.push_back(desc(data[3 * e + 2]));
    }
    const int64_t layer = exl3_moe_cpu_make_layer_raw(gate.data(), up.data(), down.data(), mixedk::E, 0, 10.0f, 0);
    for (int m : {1, 4, 8}) {
        const auto suffix = "_" + std::to_string(m) + ".bin";
        const auto dir = mixedk::directory() + "/";
        const auto x = mixedk::read<uint16_t>(dir + "x" + suffix, size_t(m) * H);
        const auto sel = mixedk::read<int32_t>(dir + "sel" + suffix, m * K);
        const auto wf = mixedk::read<float>(dir + "w" + suffix, m * K);
        const auto want = mixedk::read<float>(dir + "out" + suffix, size_t(m) * H);
        std::vector<at::Half> w;
        for (float weight : wf) w.push_back(to_half(weight));
        std::vector<float> initial;
        for (int move = 0; move < 2; ++move) {
            // Each projection keeps its own rate when its backing bytes move.
            const auto& source = move ? relocated : data;
            for (int e = 0; e < mixedk::E; ++e) {
                const auto g = desc(source[3 * e]), u = desc(source[3 * e + 1]), d = desc(source[3 * e + 2]);
                exl3_moe_cpu_set_expert_raw(layer, e, &g, &u, &d, 0);
            }
            std::vector<float> out(want.size(), std::nanf(""));
            exl3_moe_cpu_forward_raw(layer, reinterpret_cast<const at::Half*>(x.data()), sel.data(),
                                     w.data(), out.data(), m, K, threads);
            double num = 0, den = 0;
            bool finite = true;
            for (size_t i = 0; i < out.size(); ++i) {
                finite &= std::isfinite(out[i]);
                num += (double(out[i]) - want[i]) * (double(out[i]) - want[i]);
                den += double(want[i]) * want[i];
            }
            const double err = std::sqrt(num / std::max(den, 1e-300));
            std::printf("mixed K1..K6 m=%d relocated=%d rel_l2=%.6g\n", m, move, err);
            check(finite && err <= kMaxErr, "mixed K: relative L2 above 0.0325 against LinearEXL3");
            if (!move) initial = out;
            else check(out == initial, "mixed K: relocation changed output");
        }
    }
    exl3_moe_cpu_free_layer(layer);
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
    // one layer of le.size() experts whose bytes start at bytes_of(i) (the RAM copy, or the mapped experts.bin)
    auto make_layer = [&](auto bytes_of) {
        std::vector<MoeCpuMatrixDesc> g(le.size()), u(le.size()), d(le.size());
        for (size_t i = 0; i < le.size(); ++i) {
            const auto& s = pack.expert(le[i].first, le[i].second);
            const uint8_t* b = bytes_of(i);
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
        return exl3_moe_cpu_make_layer_raw(g.data(), u.data(), d.data(), (int) le.size(), 0, 10.0f, 0);
    };
    // Accuracy at m = 1, 4, 8 and the median time at m = 1, 8. `condition` runs before every timed forward and returns
    // false when it could not put the weights into the wanted state; such a mode is reported INVALID, not timed.
    // Every forward's output, timed or not, is checked (finite, error limit) outside the timer.
    struct Stats { bool valid = true; double max_bad = 0; std::vector<double> read_mb; };
    auto measure = [&](int64_t layer, const std::string& tag, auto condition, int reps, double& t1, double& t8,
                       Stats& st) {
        for (int m : {1, 4, 8}) {
            const std::string s = std::to_string(m);
            const auto x = read_bin<uint16_t>(gold + "/x_" + s + ".bin", (size_t) m * H);
            const auto sel = read_bin<int32_t>(gold + "/sel_" + s + ".bin", (size_t) m * K);
            const auto wf = read_bin<float>(gold + "/w_" + s + ".bin", (size_t) m * K);
            const auto want = read_bin<float>(gold + "/out_" + s + ".bin", (size_t) m * H);
            std::vector<at::Half> w(wf.size());
            for (size_t i = 0; i < wf.size(); ++i) w[i] = to_half(wf[i]);
            std::vector<float> out((size_t) m * H);
            auto poison = [&] { std::fill(out.begin(), out.end(), std::nanf("")); };   // a forward that writes nothing fails
            auto run = [&] {
                exl3_moe_cpu_forward_raw(layer, (const at::Half*) x.data(), sel.data(), w.data(), out.data(), m, K,
                                         threads);
            };
            auto verify = [&](const char* when) {
                double num = 0, den = 0;
                bool finite = true;
                for (size_t i = 0; i < out.size(); ++i) {
                    finite = finite && std::isfinite(out[i]);
                    num += ((double) out[i] - want[i]) * ((double) out[i] - want[i]);
                    den += (double) want[i] * want[i];
                }
                const double err = std::sqrt(num / std::max(den, 1e-300));
                check(finite, tag + " m=" + s + " " + when + ": non-finite output");
                check(err <= kMaxErr, tag + " m=" + s + " " + when + ": relative L2 above the limit");
                return err;
            };
            poison();
            run();
            const double err = verify("first forward");
            std::printf("[%s] m=%d rel_l2 vs FP16 golden = %.5f\n", tag.c_str(), m, err);
            if (tag == "ram") metric(("err_m" + s).c_str(), err);
            if (m != 4) {
                std::vector<double> t;
                for (int r = 0; r < 3; ++r) { poison(); run(); verify("warm-up"); }
                for (int r = 0; r < reps && st.valid; ++r) {
                    if (!condition(st)) { st.valid = false; break; }
                    poison();
                    const double rb0 = read_bytes_mb();
                    const double a = now_us();
                    run();
                    t.push_back(now_us() - a);
                    const double rb1 = read_bytes_mb();
                    st.read_mb.push_back(rb0 < 0 || rb1 < 0 ? std::nan("") : rb1 - rb0);   // NaN: no I/O evidence
                    verify("timed");
                }
                if (!st.valid) {
                    std::printf("  [%s] INVALID: the weights could not be put into the mode's state\n", tag.c_str());
                    return;
                }
                std::sort(t.begin(), t.end());
                const double med = t[t.size() / 2];
                std::printf("  [%s] time m=%d, %d threads: %.0f us (%.0f us per expert slot)\n", tag.c_str(), m, threads,
                            med, med / (m * K));
                (m == 1 ? t1 : t8) = med;
            }
        }
    };

    // 1. the scored mode: the experts copied to RAM
    std::vector<std::vector<uint8_t>> ram(le.size());
    for (size_t i = 0; i < le.size(); ++i) {
        const auto& s = pack.expert(le[i].first, le[i].second);
        ram[i].assign(pack.expert_base() + s.offset, pack.expert_base() + s.offset + s.bytes);
#if defined(__linux__)
        // the copy mapped these pages through the pack's own mapping; unmap them there, or a cold mode cannot drop them
        const uintptr_t pa = (uintptr_t) (pack.expert_base() + s.offset) & ~(uintptr_t) 4095;
        madvise((void*) pa, (uintptr_t) (pack.expert_base() + s.offset + s.bytes) - pa, MADV_DONTNEED);
#endif
    }
    const int64_t layer = make_layer([&](size_t i) { return (const uint8_t*) ram[i].data(); });
    double t1 = 0, t8 = 0;
    {
        Stats st;
        measure(layer, "ram", [](Stats&) { return true; }, 21, t1, t8, st);
    }

    // 2. diagnostics, not scored (K11_WEIGHTS = mmap-warm | mmap-ptecold | mmap-cold | all). The test maps
    // experts.bin itself (read-only, shared, as the engine's file tier) so that it can replace the mapping in place.
    const char* we = std::getenv("K11_WEIGHTS");
    const std::string modes = we ? we : "ram";
    if (modes != "ram") {
#if defined(__linux__)
        const std::string path = pack_dir + "/experts.bin";
        const int fd = open(path.c_str(), O_RDONLY);
        struct stat sb {};
        if (fd < 0 || fstat(fd, &sb) != 0) {
            std::printf("RESULT fail cannot-open=%s\n", path.c_str());
            return 1;
        }
        const size_t file_bytes = (size_t) sb.st_size;
        uint8_t* map = (uint8_t*) mmap(nullptr, file_bytes, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) { std::printf("RESULT fail mmap-failed\n"); return 1; }
        const int64_t flayer =
            make_layer([&](size_t i) { return (const uint8_t*) map + pack.expert(le[i].first, le[i].second).offset; });
        // a fresh mapping at the same address: every page-table entry of the old one is gone, the descriptors stay valid.
        // A failed MAP_FIXED may already have removed the old mapping: then the range is no longer ours, and the
        // remaining file modes are abandoned without touching it again (no forward, no munmap).
        bool map_lost = false;
        auto remap = [&] {
            if (map_lost) return false;
            if (mmap(map, file_bytes, PROT_READ, MAP_SHARED | MAP_FIXED, fd, 0) == (void*) map) return true;
            map_lost = true;
            std::printf("  MAP_FIXED failed: the mapping is abandoned, the remaining file modes are not run\n");
            return false;
        };
        // the 32 experts' page ranges, as (address in the mapping, length, file offset)
        auto each_range = [&](auto fn) {
            bool ok = true;
            for (const auto& [l, e] : le) {
                const auto& s = pack.expert(l, e);
                const uint64_t a = s.offset & ~(uint64_t) 4095, b = s.offset + s.bytes;
                ok = fn(map + a, (size_t) (b - a), (off_t) a) && ok;
            }
            return ok;
        };
        // fraction of the experts' pages in the page cache (mincore reports the cache for a file mapping), -1 on error
        auto resident = [&] {
            size_t in = 0, all = 0;
            std::vector<unsigned char> v;
            const bool ok = each_range([&](uint8_t* a, size_t n, off_t) {
                v.resize((n + 4095) / 4096);
                if (mincore(a, n, v.data()) != 0) return false;
                for (unsigned char c : v) { in += c & 1; ++all; }
                return true;
            });
            return ok && all ? (double) in / all : -1.0;
        };
        auto warm_ok = [&](Stats& st) {   // the page cache must hold every page
            const double r = resident();
            st.max_bad = std::max(st.max_bad, r < 0 ? 1.0 : 1.0 - r);
            return r == 1.0;
        };
        auto drop = [&] {
            return remap() && each_range([&](uint8_t*, size_t n, off_t off) {
                       return posix_fadvise(fd, off, (off_t) n, POSIX_FADV_DONTNEED) == 0;
                   });
        };
        auto cold_ok = [&](Stats& st) {   // none of the pages may be cached; up to three drops
            double r = -1;
            for (int i = 0; i < 3; ++i) {
                if (!drop()) return false;
                r = resident();
                if (r == 0.0) break;
            }
            st.max_bad = std::max(st.max_bad, r < 0 ? 1.0 : r);
            return r >= 0 && r <= 0.001;
        };
        for (const std::string mode : {"mmap-warm", "mmap-ptecold", "mmap-cold"}) {
            if (modes != "all" && modes != mode) continue;
            if (map_lost) {
                std::printf("  [%s] INVALID: not run, the mapping was lost\n", mode.c_str());
                continue;
            }
            double f1 = 0, f8 = 0;
            Stats st;
            if (mode == "mmap-warm") measure(flayer, mode, warm_ok, 21, f1, f8, st);
            else if (mode == "mmap-ptecold")
                measure(flayer, mode, [&](Stats& x) { return remap() && warm_ok(x); }, 21, f1, f8, st);
            else measure(flayer, mode, cold_ok, 9, f1, f8, st);
            bool io_known = !st.read_mb.empty();
            for (double v : st.read_mb) io_known = io_known && !std::isnan(v);
            if (io_known) std::sort(st.read_mb.begin(), st.read_mb.end());   // NaN breaks the sort's ordering contract
            const double rmb = io_known ? st.read_mb[st.read_mb.size() / 2] : std::nan("");
            char io[64];
            if (io_known) std::snprintf(io, sizeof io, "%.1f MB", rmb);
            else std::snprintf(io, sizeof io, "unavailable");
            std::printf("  [%s] %s; worst %s %.4f; median storage read per timed forward (m=1 and m=8 together) %s\n",
                        mode.c_str(), st.valid ? "valid" : "INVALID",
                        mode == "mmap-cold" ? "resident fraction" : "missing fraction", st.max_bad, io);
            if (!st.valid) continue;
            metric(("us_m1_" + mode).c_str(), f1);
            metric(("us_m8_" + mode).c_str(), f8);
            if (io_known) metric(("read_mb_" + mode).c_str(), rmb);
        }
        exl3_moe_cpu_free_layer(flayer);
        if (!map_lost) munmap(map, file_bytes);
        close(fd);
#else
        std::printf("K11_WEIGHTS=%s: file modes need Linux; skipped\n", modes.c_str());
#endif
    }
    exl3_moe_cpu_free_layer(layer);
    mixed_k(threads);
    metric("us_m1", t1);
    metric("us_m8", t8);
    metric("score_us", t1 + t8 / 8);
    std::printf("RESULT %s%s\n", failures ? "fail" : "pass", metrics.c_str());
    return failures ? 1 : 0;
}
