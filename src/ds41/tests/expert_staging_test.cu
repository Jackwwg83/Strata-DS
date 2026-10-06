// Exact blob copies, including tails, changed device counts, and two forks per graph.
#include "bench_util.hpp"
#include "strata/ds41/expert_staging.hpp"

using namespace ds41test;
namespace sd = strata::ds41;

int main() {
    require_gpu();
    Verdict v;
    constexpr size_t largest = 22151168; // At least 21.1 MiB, with aligned source slots.
    constexpr int slots = 6;
    sd::ExpertStaging stage(slots, largest);
    uint8_t *host = nullptr, *alias = nullptr;
    ck(cudaHostAlloc(&host, largest * slots, cudaHostAllocMapped), "sources");
    ck(cudaHostGetDevicePointer(&alias, host, 0), "source alias");
    for (size_t i = 0; i < largest * slots; ++i) host[i] = uint8_t((i * 13) ^ (i >> 11));
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
    cudaFreeHost(host);
    return v.finish();
}
