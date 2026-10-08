// src/ds41/tests/stage_bench.cu - throughput of the staging copy (ExpertStaging) from mapped host memory, as a decode
// step uses it: 40 layers x 2 jobs of one expert's size (7.1 MB, the SAGE 1.59bpw pack), from random places in a
// large host region (the RAM tier). One line per launch shape (DS41_STAGE_BLOCKS / DS41_STAGE_UNROLL) and host
// page size (4 KiB, or 2 MiB transparent huge pages as DS41_RAM_HUGEPAGES=1 asks for).
//   stage_bench [--host-gib 8] [--rounds 20]
#include "bench_util.hpp"
#include "strata/ds41/expert_staging.hpp"

#include <sys/mman.h>

#include <chrono>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace ds41test;
namespace sd = strata::ds41;

int main(int argc, char** argv) {
    require_gpu();
    double host_gib = 8;
    int rounds = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--host-gib") host_gib = std::stod(argv[i + 1]);
        else if (a == "--rounds") rounds = std::stoi(argv[i + 1]);
    }
    constexpr size_t kExpert = 7438336;   // 7.1 MiB, 4 KiB aligned
    constexpr int kLayers = 40, kJobs = 2;
    const size_t host_bytes = (size_t) (host_gib * (1ull << 30)) / kExpert * kExpert;
    const size_t n_slots = host_bytes / kExpert;
    for (bool huge : {false, true}) {
        constexpr size_t kHuge = 2u << 20;
        void* map = mmap(nullptr, host_bytes + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (map == MAP_FAILED) { std::printf("cannot map %zu bytes\n", host_bytes); return 1; }
        auto* host = (uint8_t*) (((uintptr_t) map + kHuge - 1) & ~(uintptr_t) (kHuge - 1));
        madvise(host, host_bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
        std::memset(host, 0x5a, host_bytes);   // touch: the pages exist before they are pinned
        ck(cudaHostRegister(host, host_bytes, cudaHostRegisterMapped), "register");
        uint8_t* alias = nullptr;
        ck(cudaHostGetDevicePointer(&alias, host, 0), "alias");
        cudaStream_t st;
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        for (auto [blocks, unroll] : {std::pair{68, 1}, {68, 2}, {68, 4}, {68, 8}, {136, 4}, {170, 4}, {340, 4}}) {
            sd::ExpertStaging stage(kJobs, kExpert);
            stage.set_launch(blocks, unroll);
            std::mt19937 g(7);
            // a step's job lists, uploaded before timing
            std::vector<std::vector<sd::ExpertCopy>> lists(kLayers);
            for (auto& l : lists)
                for (int j = 0; j < kJobs; ++j)
                    l.push_back({alias + (g() % n_slots) * kExpert, stage.data() + j * stage.stride(), kExpert});
            Dev<sd::ExpertCopy> jobs(kLayers * kJobs);
            for (int l = 0; l < kLayers; ++l)
                ck(cudaMemcpy(jobs.p + l * kJobs, lists[l].data(), kJobs * sizeof(sd::ExpertCopy),
                              cudaMemcpyHostToDevice), "jobs");
            const int count = kJobs;
            ck(cudaMemcpy(stage.count(), &count, sizeof(int), cudaMemcpyHostToDevice), "count");
            double best = 1e30;
            for (int r = 0; r < rounds; ++r) {
                ck(cudaStreamSynchronize(st), "idle");
                const auto t0 = std::chrono::steady_clock::now();
                for (int l = 0; l < kLayers; ++l) {   // a layer's jobs, then the copy, as publish + fork_copy do
                    ck(cudaMemcpyAsync(stage.jobs(), jobs.p + l * kJobs, kJobs * sizeof(sd::ExpertCopy),
                                       cudaMemcpyDeviceToDevice, st), "layer jobs");
                    stage.fork_copy(st);
                    stage.join(st);
                }
                ck(cudaStreamSynchronize(st), "step");
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            const double bytes = (double) kLayers * kJobs * kExpert;
            std::printf("pages %-4s blocks %3d unroll %d: %.2f ms per step of %d experts, %.1f GB/s\n",
                        huge ? "2M" : "4K", blocks, unroll, best * 1e3, kLayers * kJobs, bytes / best / 1e9);
        }
        {   // the same bytes by the copy engine (cudaMemcpyAsync from the pinned region): the link's DMA rate
            Dev<uint8_t> dst(kJobs * kExpert);
            std::mt19937 g(7);
            double best = 1e30;
            for (int r = 0; r < rounds; ++r) {
                ck(cudaStreamSynchronize(st), "idle");
                const auto t0 = std::chrono::steady_clock::now();
                for (int l = 0; l < kLayers; ++l)
                    for (int j = 0; j < kJobs; ++j)
                        ck(cudaMemcpyAsync(dst.p + j * kExpert, host + (g() % n_slots) * kExpert, kExpert,
                                           cudaMemcpyHostToDevice, st), "dma");
                ck(cudaStreamSynchronize(st), "step");
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            const double bytes = (double) kLayers * kJobs * kExpert;
            std::printf("pages %-4s copy engine (cudaMemcpyAsync):   %.2f ms per step, %.1f GB/s\n", huge ? "2M" : "4K",
                        best * 1e3, bytes / best / 1e9);
        }
        cudaStreamDestroy(st);
        cudaHostUnregister(host);
        munmap(map, host_bytes + kHuge);
    }
    return 0;
}
