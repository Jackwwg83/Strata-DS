// Run with the production CPU layer on Linux, or the extracted host harness.
#include "moe_mul1.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

static std::atomic<long> live{0};
static long fail_after = -1;
void* operator new(size_t n) {
    if (fail_after == 0) throw std::bad_alloc();
    if (fail_after > 0) --fail_after;
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    ++live;
    return p;
}
void operator delete(void* p) noexcept { if (p) { --live; std::free(p); } }
void* operator new[](size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
#if defined(__cpp_sized_deallocation)
void operator delete(void* p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, size_t) noexcept { ::operator delete(p); }
#endif

static int failures = 0;
static void check(bool ok, const char* msg) {
    if (!ok) { ++failures; std::printf("FAIL %s\n", msg); }
}
struct Weights {
    std::vector<uint16_t> bytes[3][3];
    at::Half scale[256]{};
    MoeCpuMatrixDesc m[3][3];
    Weights() {
        for (int p = 0; p < 3; ++p) for (int e = 0; e < 3; ++e) {
            const int rate = e == 0 ? 1 : (e == 1 ? 6 : p + 2);
            bytes[p][e].resize(8 * 8 * 16 * rate, uint16_t(0x1100 + p * 16 + e));
            m[p][e] = {bytes[p][e].data(), scale, scale, 8, 8, 16 * rate};
        }
    }
    int64_t make(bool gated = true, int swz = 0) {
        return exl3_moe_cpu_make_layer_raw(gated ? m[0] : nullptr, m[1], m[2], 3, gated ? 0 : 2, 10.f, swz);
    }
};
static void staging(Weights& w) {
    for (bool gated : {false, true}) for (int swz : {0, 1}) {
        for (bool reverse : {false, true}) {
            if (reverse) for (int p = 0; p < 3; ++p) std::swap(w.m[p][0], w.m[p][1]);
            const auto h = w.make(gated, swz);
            for (int threads : {1, 4}) for (auto ids : {std::vector<uint32_t>{1}, {2, 1, 0, 2}}) {
                std::vector<uint8_t> want;
                for (auto e : ids) for (int p = gated ? 0 : 1; p < 3; ++p) {
                    const auto& d = w.m[p][e];
                    const auto* b = reinterpret_cast<const uint8_t*>(d.trellis);
                    want.insert(want.end(), b, b + 8 * 8 * d.tile_w * 2);
                }
                // Extra guard space lets the old overflow fail without corrupting the heap.
                std::vector<uint8_t> dst(64 + std::max<size_t>(want.size(), ids.size() * 3 * 12288) + 64, 0xCD);
                exl3_moe_cpu_stage_experts(h, ids.data(), int(ids.size()), dst.data() + 64, threads);
                check(std::equal(want.begin(), want.end(), dst.begin() + 64), "mixed-K staged bytes");
                check(std::all_of(dst.begin(), dst.begin() + 64, [](auto x) { return x == 0xCD; }), "prefix guard");
                check(std::all_of(dst.begin() + 64 + want.size(), dst.end(), [](auto x) { return x == 0xCD; }), "suffix guard");
            }
            exl3_moe_cpu_free_layer(h);
            if (reverse) for (int p = 0; p < 3; ++p) std::swap(w.m[p][0], w.m[p][1]);
        }
    }
}
static void shapes(Weights& w) {
    for (bool gated : {false, true}) for (int p = gated ? 0 : 1; p < 3; ++p)
        for (int e = 0; e < 3; ++e) for (bool k : {false, true})
            for (int bad : {0, -8, 16, std::numeric_limits<int>::max()}) {
                auto& dim = k ? w.m[p][e].k_tiles : w.m[p][e].n_tiles;
                const int saved = dim;
                dim = bad;
                bool refused = false;
                try { exl3_moe_cpu_free_layer(w.make(gated)); }
                catch (const std::runtime_error&) { refused = true; }
                dim = saved;
                check(refused, "bad projection shape must fail");
            }
    for (bool gated : {false, true}) exl3_moe_cpu_free_layer(w.make(gated));
}
int main(int argc, char** argv) {
    if (argc < 2) return 2;
    Weights w;
    if (!std::strcmp(argv[1], "stage")) staging(w);
    else if (!std::strcmp(argv[1], "shape")) shapes(w);
    else if (!std::strcmp(argv[1], "leak")) {
        for (int fault = 0; fault < 4; ++fault) {
            const auto gate = w.m[0][0], down = w.m[2][2];
            if (fault == 0) w.m[2][2].tile_w = 0;
            if (fault == 1) w.m[2][2].trellis = nullptr;
            if (fault == 2) w.m[2][2].n_tiles = 16;
            if (fault == 3) w.m[0][0].k_tiles = 16;
            const auto before = live.load();
            for (int i = 0; i < 16; ++i) {
                bool refused = false;
                try { exl3_moe_cpu_free_layer(w.make()); }
                catch (const std::runtime_error&) { refused = true; }
                check(refused, "invalid descriptor or shape must fail");
            }
            check(live == before, "failed registration must release all allocations");
            w.m[0][0] = gate;
            w.m[2][2] = down;
        }
    } else if (!std::strcmp(argv[1], "alloc") && argc == 3) {
        const auto before = live.load();
        fail_after = std::strtol(argv[2], nullptr, 10);
        bool refused = false;
        int64_t h = -1;
        try { h = w.make(); } catch (const std::bad_alloc&) { refused = true; }
        fail_after = -1;
        if (refused) check(live == before, "allocation failure must release all allocations");
        else { exl3_moe_cpu_free_layer(h); std::puts("registration reached success"); }
    } else return 2;
    std::printf("RESULT %s %s (%d failures)\n", failures ? "fail" : "pass", argv[1], failures);
    return failures ? 1 : 0;
}
