// Supplemental raw-forward concurrency/guard test. Link separately with the control
// and candidate vendor objects; compare the output files for exact numerical parity.
// No model pack or CUDA runtime is needed. Set K11_FULL_CONCURRENT=1 for full shapes.
#include "moe_mul1.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <thread>

static at::Half bits(uint16_t value) { return at::Half(value, at::Half::from_bits()); }

struct OwnedMatrix
{
    std::vector<uint16_t> packed;
    std::vector<at::Half> suh, svh;
    MoeCpuMatrixDesc desc;
    OwnedMatrix(int k, int n, int words, std::mt19937& rng)
        : packed(size_t(k / 16) * (n / 16) * words), suh(k), svh(n)
    {
        for (auto& x : packed) x = uint16_t(rng());
        for (auto& x : suh) x = bits(uint16_t(0x3800 | (rng() & 0x83ff)));
        for (auto& x : svh) x = bits(uint16_t(0x2000 | (rng() & 0x83ff)));
        desc = {packed.data(), suh.data(), svh.data(), k / 16, n / 16, words};
    }
};

struct Job
{
    int h, f;
    int64_t layer;
    std::vector<OwnedMatrix> matrices;
    std::vector<MoeCpuMatrixDesc> gates, ups, downs;
    std::vector<at::Half> input, weights;
    std::vector<int32_t> selected;
    std::vector<std::vector<float>> reference;
    static constexpr int token_counts[4] = {1, 4, 8, 9};

    Job(int seed, bool wide, bool full) : h(full ? 5120 : 256), f(full ? 2304 : 128)
    {
        constexpr int experts = 8;
        const int rates[experts] = {48, 48, 48, 24, 40, 56, 64, 128};
        std::mt19937 rng(seed);
        matrices.reserve(experts * 3);
        for (int e = 0; e < experts; ++e)
        {
            matrices.emplace_back(h, f, rates[e], rng); gates.push_back(matrices.back().desc);
            matrices.emplace_back(h, f, rates[e], rng); ups.push_back(matrices.back().desc);
            matrices.emplace_back(f, h, rates[e], rng); downs.push_back(matrices.back().desc);
        }
        layer = exl3_moe_cpu_make_layer_raw(gates.data(), ups.data(), downs.data(), experts, 0, 10.f, 0);
        input.resize(9 * h); weights.resize(9 * 6); selected.resize(9 * 6);
        for (auto& x : input) x = bits(uint16_t((wide ? 0x0400 : 0x3c00) | (rng() & 0x83ff)));
        for (int t = 0; t < 9; ++t)
        {
            if (wide) input[t * h] = bits(0x5800); // 128 plus small residuals
            for (int j = 0; j < 6; ++j)
            {
                weights[t * 6 + j] = bits(uint16_t(0x3155 | ((j & 1) ? 0x8000 : 0)));
                selected[t * 6 + j] = wide ? j : (t + j) % experts;
            }
        }
        for (int m : token_counts)
        {
            reference.emplace_back(m * h);
            exl3_moe_cpu_forward_raw(layer, input.data(), selected.data(), weights.data(),
                                    reference.back().data(), m, 6, 1);
        }
    }
};

int main(int argc, char** argv)
{
    if (argc != 2) { std::fprintf(stderr, "usage: %s output.bin\n", argv[0]); return 2; }
    const bool full = std::getenv("K11_FULL_CONCURRENT") != nullptr;
    Job a(71267, false, full), b(92113, true, full);
    std::atomic<int> ready{0}, failures{0};
    std::atomic<bool> go{false};
    auto run = [&](Job& job, int offset)
    {
        ++ready;
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        const int thread_counts[7] = {1, 3, 8, 12, 16, 24, 32};
        for (int iteration = 0; iteration < 28; ++iteration)
        {
            const int index = (iteration + offset) % 4;
            const int m = Job::token_counts[index];
            std::vector<float> output(m * job.h);
            exl3_moe_cpu_forward_raw(job.layer, job.input.data(), job.selected.data(), job.weights.data(),
                                    output.data(), m, 6, thread_counts[(iteration + offset) % 7]);
            if (std::memcmp(output.data(), job.reference[index].data(), output.size() * sizeof(float)))
                ++failures;
        }
    };
    std::thread first(run, std::ref(a), 0), second(run, std::ref(b), 3);
    while (ready.load() != 2) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    first.join(); second.join();
    std::ofstream dump(argv[1], std::ios::binary);
    size_t values = 0;
    for (const Job* job : {&a, &b})
        for (const auto& output : job->reference)
        {
            for (float value : output) if (!std::isfinite(value)) ++failures;
            dump.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
            values += output.size();
        }
    const bool wrote = bool(dump);
    dump.close();
    std::printf("%s concurrent_hosts=2 calls=56 rows=1,4,8,9 threads=1,3,8,12,16,24,32 full=%d values=%zu failures=%d\n",
                failures || !wrote ? "FAIL" : "PASS", full ? 1 : 0, values, int(failures));
    std::fflush(stdout);
    // Pool helpers are intentionally process-lifetime; avoid their static teardown race.
    std::_Exit(failures || !wrote ? 1 : 0);
}
