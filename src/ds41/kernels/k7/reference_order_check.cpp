// CPU arithmetic regression for k7_hc.cu, independent of CUDA availability.
// Run from the repository root:
//   c++ -O2 -std=c++17 -ffp-contract=off src/ds41/kernels/k7/reference_order_check.cpp -o /tmp/k7_order_check
//   /tmp/k7_order_check
// This checks the reduction mapping, not GPU execution, expf/rsqrt accuracy,
// races, graph capture, acceptance-test results, or performance.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int kN = 20480;
using Input = std::array<float, kN>;

float bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    bits &= 0xffff0000;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float warp_sum(std::array<float, 32> values) {
    for (int offset = 16; offset; offset /= 2) {
        const auto old = values;
        for (int lane = 0; lane + offset < 32; ++lane)
            values[lane] = old[lane] + old[lane + offset];
    }
    return values[0];
}

// ops.cu: one complete stride-256 dot or stride-1024 norm chain per
// reference thread, five shuffle levels, then thread-zero warp-order sum.
float reference(const Input& x, const Input& w, bool norm) {
    const int threads = norm ? 1024 : 256;
    std::vector<float> lanes(threads);
    for (int tid = 0; tid < threads; ++tid)
        for (int column = tid; column < kN; column += threads)
            lanes[tid] = std::fma(x[column], norm ? x[column] : w[column], lanes[tid]);
    float result = 0;
    for (int warp = 0; warp < threads / 32; ++warp) {
        std::array<float, 32> values;
        std::copy_n(lanes.begin() + warp * 32, 32, values.begin());
        result += warp_sum(values);
    }
    return result;
}

// Emulate the revised two-row shared-tile producer and ordered consumer.
// Register chains remain live across all five 16-step cache tiles.
float candidate(const Input& x, const Input& w, bool norm) {
    std::array<float, 32> totals{};
    for (int reference_warp = 0; reference_warp < 8; ++reference_warp) {
        if (norm) {
            for (int row_warp = 0; row_warp < 2; ++row_warp) {
                for (int q = 0; q < 2; ++q) {
                    std::array<float, 32> values{};
                    for (int first = 0; first < 80; first += 16) {
                        for (int step = 0; step < 16; ++step) {
                            if ((step & 3) != row_warp + 2 * q) continue;
                            for (int lane = 0; lane < 32; ++lane) {
                                const int column = reference_warp * 32 + lane + 256 * (first + step);
                                values[lane] = std::fma(x[column], x[column], values[lane]);
                            }
                        }
                    }
                    totals[(row_warp + 2 * q) * 8 + reference_warp] = warp_sum(values);
                }
            }
        } else {
            std::array<float, 32> values{};
            for (int first = 0; first < 80; first += 16) {
                std::array<float, 512> cache_x, cache_w;
                for (int tid = 0; tid < 64; ++tid) {
                    for (int local = tid; local < 512; local += 64) {
                        const int column = reference_warp * 32 + (first + local / 32) * 256 + local % 32;
                        cache_x[local] = x[column];
                    }
                }
                for (int lane = 0; lane < 32; ++lane) {
                    for (int local = lane * 4; local < 512; local += 128) {
                        const int column = reference_warp * 32 + (first + local / 32) * 256 + local % 32;
                        for (int component = 0; component < 4; ++component)
                            cache_w[local + component] = w[column + component];
                    }
                }
                for (int step = 0; step < 16; ++step) {
                    for (int lane = 0; lane < 32; ++lane) {
                        const int local = step * 32 + lane;
                        values[lane] = std::fma(cache_x[local], cache_w[local], values[lane]);
                    }
                }
            }
            totals[reference_warp] = warp_sum(values);
        }
    }
    float result = 0;
    for (int warp = 0; warp < (norm ? 32 : 8); ++warp) result += totals[warp];
    return result;
}

bool equal_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

bool check(const Input& x, const Input& w, const char* label) {
    for (bool norm : {false, true}) {
        const float expected = reference(x, w, norm);
        const float actual = candidate(x, w, norm);
        if (!equal_bits(expected, actual)) {
            std::printf("FAIL %s %s: reference=%.9g candidate=%.9g\n",
                        label, norm ? "RMS" : "dot", expected, actual);
            return false;
        }
    }
    return true;
}
}  // namespace

int main() {
    Input x, w{};
    x.fill(1);
    w[0] = 33554432;
    w[1024] = -33554432;
    w[1280] = 1;
    if (!check(x, w, "reported cancellation") || candidate(x, w, false) != 1) return 1;
    const float pre0 = 1.f / (1.f + std::exp(-candidate(x, w, false))) + 1e-6f;
    std::printf("PASS reported cancellation: dot=1 pre0=%.9g\n", pre0);
    int cases = 1;

    for (int tid = 0; tid < 256; ++tid) {
        for (int pattern = 0; pattern < 3; ++pattern) {
            w.fill(0);
            const float large = std::ldexp(1.f, 25 + tid % 70);
            const int a = pattern == 0 ? 0 : pattern == 1 ? 15 : 63;
            const int b = pattern == 0 ? 4 : pattern == 1 ? 16 : 78;
            const int c = pattern == 0 ? 5 : pattern == 1 ? 17 : 79;
            w[tid + 256 * a] = large;
            w[tid + 256 * b] = -large;
            w[tid + 256 * c] = 1;
            if (!check(x, w, "all-lane boundary cancellation")) return 1;
            ++cases;
        }
    }

    std::mt19937 rng(935);
    for (int family = 0; family < 5; ++family) {
        for (int m = 1; m <= 8; ++m) {
            for (int token = 0; token < m; ++token) {
                for (int c = 0; c < kN; ++c) {
                    if (family == 0) {
                        x[c] = 1;
                        w[c] = c % 3 == 0 ? std::ldexp(1.f, c / 3 % 70) :
                               c % 3 == 1 ? -std::ldexp(1.f, c / 3 % 70) : 1;
                    } else if (family == 1) {
                        x[c] = bf16(std::ldexp(float(int(rng() % 7) - 3), int(rng() % 31) - 15));
                        w[c] = std::ldexp(float(int(rng() % 7) - 3), int(rng() % 71) - 35);
                    } else if (family == 2) {
                        x[c] = bf16((c % 2 ? -1.f : 1.f) * std::ldexp(1.f, c / 256 % 30 - 15));
                        w[c] = std::ldexp(float(int(rng() % 3) - 1), 60);
                    } else if (family == 3) {
                        x[c] = bf16(std::ldexp(1.f, int(rng() % 81) - 40));
                        w[c] = std::ldexp(float(int(rng() % 7) - 3), int(rng() % 41) - 20);
                    } else {
                        x[c] = bf16(float(int(rng() % 9) - 4) * 1e-16f);
                        w[c] = std::ldexp(float(int(rng() % 7) - 3), int(rng() % 61));
                    }
                }
                if (!check(x, w, "dynamic range, RMS and tiny BF16")) return 1;
                ++cases;
            }
        }
    }
    std::printf("PASS %d CPU dot/RMS cases bitwise match reference order; all m=1..8 covered\n", cases);
    return 0;
}
