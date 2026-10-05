// CPU support for check_output_host.py. The checker inserts the ACTUAL helper
// and output-kernel bodies below; no independently reimplemented Hadamard.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define __device__
#define __global__
using half = _Float16;  // IEEE binary16 host STORAGE and exact conversion to float
struct half2 { half x, y; };
struct alignas(8) half4 { half2 x, y; };
struct alignas(16) float4 { float x, y, z, w; };
struct Dim { int x = 0, y = 0; } threadIdx, blockIdx;
static float __low2float(half2 h) { return float(h.x); }
static float __high2float(half2 h) { return float(h.y); }
static uint32_t __float_as_uint(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
static float __uint_as_float(uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }
static float __fadd_rn(float a, float b) { return a + b; }

// Replay each lane until its next shuffle, then resolve the whole warp's
// simultaneous exchange. Each completed shuffle holds a snapshot. No threads
// or simulated races: only final per-lane stores commit, once, at completion.
struct AwaitShuffle {};
struct Stage { std::array<uint64_t, 32> values{}; int delta = -1; };
static std::vector<Stage> stages;
static Stage pending;
static size_t cursor;
static uint64_t __shfl_xor_sync(unsigned mask, uint64_t v, int delta) {
    assert(mask == 0xffffffffu && delta > 0 && delta < 32);
    if (cursor < stages.size()) {
        const auto& s = stages[cursor++];
        assert(s.delta == delta);
        return s.values[threadIdx.x ^ delta];
    }
    assert(pending.delta == -1 || pending.delta == delta);
    pending.delta = delta;
    pending.values[threadIdx.x] = v;
    throw AwaitShuffle{};
}
template<class F> static void warp(F fn) {
    stages.clear();
    for (;;) {
        pending = Stage{};
        int waiting = 0;
        for (int lane = 0; lane < 32; ++lane) {
            threadIdx.x = lane;
            cursor = 0;
            try { fn(lane); } catch (AwaitShuffle&) { ++waiting; }
        }
        if (!waiting) break;
        assert(waiting == 32);  // every slot decision must be warp-uniform
        stages.push_back(pending);
        assert(stages.size() < 2000);
    }
}

// ACTUAL_HELPERS

constexpr int H = 5120;
constexpr float HAD_SCALE = 0.088388347648f;
struct Exl3Proj { const uint16_t* trellis; const half* suh; const half* svh; int k, n, tile_w; };
struct Exl3Expert { Exl3Proj w1, w3, w2; };

// ACTUAL_OUTPUT_KERNEL

static uint32_t rng = 0x914fec43u;
static uint32_t next() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float sample() {
    // Mixed magnitudes, signs, zeros and subnormal inputs; avoid overflow so
    // bitwise equality does not depend on host-specific NaN payload selection.
    const int exponent = int(next() % 174) - 149;
    return std::ldexp(float(int(next() % 2049) - 1024), exponent);
}
static bool equal(float a, float b) { return __float_as_uint(a) == __float_as_uint(b); }
static void equal_vector(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if (!equal(a[i], b[i])) {
            std::fprintf(stderr, "different column %zu: 0x%08x vs 0x%08x\n", i,
                         __float_as_uint(a[i]), __float_as_uint(b[i]));
            std::abort();
        }
    }
}

int main() {
    // Mapping proof over every column, including the final chunk and lane.
    std::array<int, H> seen{};
    for (int chunk = 0; chunk < H / 128; ++chunk)
        for (int lane = 0; lane < 32; ++lane)
            for (int i = 0; i < 4; ++i) ++seen[chunk * 128 + 4 * lane + i];
    for (int n : seen) assert(n == 1);
    std::puts("PASS output ownership: all 5120 columns exactly once, last column 5119");

    alignas(16) std::array<float, 128> input{}, old{}, got{};
    std::vector<half> scales(H);
    size_t helper_cases = 0;
    for (int test = 0; test < 240; ++test) {
        blockIdx.y = test % (H / 128);
        for (float& f : input) f = sample();
        if (test < 128) { input.fill(0.0f); input[test] = 1.0f; }
        for (half& h : scales) h = half(std::ldexp(float(int(next() % 65) - 32), -4));
        warp([&](int) { had_ff_r_128_inner<false, true>(input.data(), old.data(), scales.data(), HAD_SCALE); });
        warp([&](int lane) { const auto v = had_ff_r_128_registers<false, true>(input.data(), scales.data(), HAD_SCALE);
                            got[4*lane] = v.x; got[4*lane+1] = v.y; got[4*lane+2] = v.z; got[4*lane+3] = v.w; });
        for (int i = 0; i < 128; ++i) assert(equal(old[i], got[i]));
        ++helper_cases;
    }
    std::printf("PASS %zu actual-helper warp comparisons: all 128 basis vectors, signs, mixed magnitudes, all scale chunks\n", helper_cases);

    size_t cases = 0, active = 0;
    std::array<std::vector<half>, 3> svh;
    std::array<Exl3Expert, 3> experts{};
    for (int e = 0; e < 3; ++e) {
        svh[e].resize(H);
        for (int i = 0; i < H; ++i) svh[e][i] = half(std::ldexp(float((i * 3 + e * 7) % 43 - 21), -4));
        experts[e].w2.svh = svh[e].data();
    }
    for (int out_offset : {0, 4, 8, 12, 16, 20, 24, 28})
    for (int m : {1, 4, 8}) for (int topk : {1, 2, 6, 7, 17, 32}) {
        const int masks = topk == 6 ? 64 : 5;
        for (int mask = 0; mask < masks; ++mask) {
            const int token = mask % m;
            const int chunk = (mask * 13 + topk * 7) % (H / 128);
            blockIdx.x = token;
            blockIdx.y = chunk;
            std::vector<int32_t> sel(m * topk, -1);
            // Poison all skipped scratch. Any accidental empty-slot read
            // contaminates the candidate and fails exact comparison.
            std::vector<float> down(size_t(m) * topk * H, std::numeric_limits<float>::quiet_NaN());
            std::vector<float> initial(size_t(m) * H + 32);
            for (float& f : initial) f = sample();
            if (mask == 0) for (size_t i = 0; i < initial.size(); ++i)
                initial[i] = __uint_as_float(i & 1 ? 0x80000000u : 0u);
            auto want = initial;
            // Overallocate only the harness's storage, then place the logical
            // output at each natural float offset modulo32. Guard both ends.
            const float guard = __uint_as_float(0x4bac9731u);
            std::vector<float> have(initial.size() + 24, guard);
            const size_t pad = ((32 - (reinterpret_cast<uintptr_t>(have.data()) & 31u)) & 31u) / 4 +
                               size_t(out_offset / 4) + 8;
            float* have_out = have.data() + pad;
            assert((reinterpret_cast<uintptr_t>(have_out) & 31u) == unsigned(out_offset));
            std::copy(initial.begin(), initial.end(), have_out);
            for (int j = 0; j < topk; ++j) {
                const bool live = topk == 6 ? ((mask >> j) & 1) :
                    (mask == 0 ? false : mask == 1 ? true : mask == 2 ? j == topk - 1 :
                     mask == 3 ? j % 2 == 0 : j % 3 != 0);
                const int slot = token * topk + j;
                sel[slot] = live ? (j + mask) % 3 : -1 - (j % 3);
                if (!live) continue;
                ++active;
                float* d = down.data() + size_t(slot) * H + chunk * 128;
                for (int i = 0; i < 128; ++i) d[i] = sample();
                warp([&](int) { had_ff_r_128_inner<false, true>(d, old.data(), experts[sel[slot]].w2.svh, HAD_SCALE); });
                for (int i = 0; i < 128; ++i) want[size_t(token) * H + chunk * 128 + i] += old[i];
            }
            warp([&](int) { output_had_add(sel.data(), topk, experts.data(), down.data(), have_out); });
            equal_vector(want, std::vector<float>(have_out, have_out + want.size()));
            for (size_t i = 0; i < pad; ++i) assert(equal(have[i], guard));
            for (size_t i = pad + initial.size(); i < have.size(); ++i) assert(equal(have[i], guard));
            ++cases;
        }
    }
    std::printf("PASS %zu actual-kernel cases / %zu live slots: m1/4/8, topk1/2/6/7/17/32, all64 six-slot masks, duplicate IDs, negative IDs, arbitrary initial output, poisoned scratch, output offsets0/4/8/12/16/20/24/28, guards on both ends\n", cases, active);

    // Witnesses make order, output preservation, and the no-FMA boundary
    // meaningful checks rather than tests which happen to be insensitive.
    float a = 33554432.0f, b = -33554432.0f, c = 1.0f;
    assert(!equal((a + b) + c, a + (b + c)));
    const float x = __uint_as_float(0x3f800001u);
    const float y = __uint_as_float(0x3f7fffffu);
    volatile float mul = x * y;
    assert(!equal(mul - 1.0f, std::fma(x, y, -1.0f)));
    std::puts("PASS sensitivity witnesses: slot reassociation and fused multiply-add change bits");
}
