// Exact blob copies, including tails, changed device counts, and two forks per graph, for every launch shape
// (blocks, loads in flight per thread) that DS41_STAGE_BLOCKS / DS41_STAGE_UNROLL can choose.
#include "bench_util.hpp"
#include "strata/ds41/expert_staging.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

using namespace ds41test;
namespace sd = strata::ds41;

namespace {

constexpr size_t largest = 22151168; // At least 21.1 MiB, with aligned source slots.
constexpr int slots = 6;

void run(Verdict& v, uint8_t* host, uint8_t* alias, int blocks, int unroll) {
    std::printf("launch: %d blocks, %d loads in flight per thread\n", blocks, unroll);
    sd::ExpertStaging stage(slots, largest);
    stage.set_launch(blocks, unroll);
    Dev<uint8_t> first(stage.stride() * slots);
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    auto enqueue = [&] {
        ck(cudaMemsetAsync(stage.data(), 0xa5, first.n, stream), "poison slots");
        stage.fork_copy(stream);
        stage.join(stream);
        ck(cudaMemcpyAsync(first.p, stage.data(), first.n, cudaMemcpyDeviceToDevice, stream), "first copy");
        ck(cudaMemsetAsync(stage.data(), 0xa5, first.n, stream), "poison reused slots");
        stage.fork_copy(stream);
        stage.join(stream);
    };
    ck(cudaMemset(stage.count(), 0, sizeof(int)), "initial count");
    ck(cudaDeviceSynchronize(), "initial inputs");
    enqueue();
    ck(cudaStreamSynchronize(stream), "warm up");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
    enqueue();
    ck(cudaStreamEndCapture(stream, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    const size_t sizes[] = {0, 1, 15, 16, 17, 255, 4095, 4097, 6800003, largest - 1, largest};
    for (size_t round = 0; round < sizeof(sizes) / sizeof(sizes[0]); ++round) {
        // Includes a zero count after nonzero counts. Unused destination bytes must stay poisoned.
        const int count = int(round % (slots + 1));
        std::vector<sd::ExpertCopy> jobs(slots);
        std::vector<uint8_t> want(first.n, 0xa5);
        for (int j = 0; j < count; ++j) {
            const size_t n = sizes[(round + j) % (sizeof(sizes) / sizeof(sizes[0]))];
            const int dst = slots - 1 - j;
            jobs[j] = {alias + j * largest, stage.data() + dst * stage.stride(), n};
            std::memcpy(want.data() + dst * stage.stride(), host + j * largest, n);
        }
        ck(cudaMemcpy(stage.jobs(), jobs.data(), jobs.size() * sizeof(jobs[0]), cudaMemcpyHostToDevice), "jobs");
        ck(cudaMemcpy(stage.count(), &count, sizeof(count), cudaMemcpyHostToDevice), "count");
        ck(cudaDeviceSynchronize(), "changed inputs");
        for (bool replay : {false, true}) {
            if (replay) ck(cudaGraphLaunch(exec, stream), "replay");
            else enqueue();
            ck(cudaStreamSynchronize(stream), "copy complete");
            const auto a = first.down();
            std::vector<uint8_t> b(first.n);
            ck(cudaMemcpy(b.data(), stage.data(), b.size(), cudaMemcpyDeviceToHost), "second result");
            v.check(a == want && b == want, "exact bytes and guards, eager and repeated graph forks");
        }
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

}  // namespace

int main() {
    require_gpu();
    Verdict v;
    v.check([] {
        sd::ExpertStaging s(1, 64);
        for (auto [b, u] : {std::pair{0, 1}, {68, 3}, {68, 16}, {-1, 4}}) {
            try {
                s.set_launch(b, u);
                return false;
            } catch (const std::invalid_argument&) {
            }
        }
        return true;
    }(), "bad launch shapes are refused");
    uint8_t *host = nullptr, *alias = nullptr;
    ck(cudaHostAlloc(&host, largest * slots, cudaHostAllocMapped), "sources");
    ck(cudaHostGetDevicePointer(&alias, host, 0), "source alias");
    for (size_t i = 0; i < largest * slots; ++i) host[i] = uint8_t((i * 13) ^ (i >> 11));
    for (auto [blocks, unroll] : {std::pair{68, 1}, {68, 4}, {170, 4}, {32, 8}, {7, 2}})
        run(v, host, alias, blocks, unroll);
    cudaFreeHost(host);
    return v.finish();
}
