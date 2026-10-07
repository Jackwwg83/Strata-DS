// src/ds41/tests/expert_stream_test.cu - prefill's expert stream on a fake pack: every job's slot holds the right
// expert's bytes when wait() returns, a slot is not overwritten before the consumer's work on it has run (the
// consumer is made slow with a spin kernel before it copies the slot out), drain() restarts the job numbers; with
// plain reads and with O_DIRECT (experts that do not start on a 4 KiB boundary).
#include "strata/ds41/expert_stream.hpp"

#include "bench_util.hpp"
#include "fake_pack.hpp"

using namespace ds41test;
namespace sd = strata::ds41;

namespace {

__global__ void spin(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

}  // namespace

int main() {
    require_gpu();
    Verdict v;
    const std::string dir = "ds41_fake_pack";   // the working directory: /tmp may refuse O_DIRECT
    write_fake_pack(dir);
    sd::Pack pack(dir);
    pack.map_experts();
    constexpr int kSlots = 8, kJobs = 300;
    const size_t slot_bytes = kExpertBytes;
    Dev<uint8_t> ring((size_t) kSlots * slot_bytes), check((size_t) kJobs * slot_bytes);
    for (int direct = 0; direct < 2; ++direct) {
    sd::ExpertStream stream(pack, nullptr, ring.p, kSlots, slot_bytes, 3, 5, direct);   // fewer host buffers than slots
    std::printf("%s\n", stream.unbuffered() ? "O_DIRECT reads" : "plain reads");
    for (int round = 0; round < 2; ++round) {
        std::vector<std::pair<int, int>> jobs;
        for (int j = 0; j < kJobs; ++j) jobs.push_back({(j * 3 + round) % L, (j * 37 + 5 * round) % E});
        // pushed in two parts, as the engine pushes layer by layer
        const int64_t first = stream.push({jobs.begin(), jobs.begin() + 100});
        stream.push({jobs.begin() + 100, jobs.end()});
        v.check(first == 0, "job numbers start at 0 (again after drain)");
        for (int j = 0; j < kJobs; ++j) {
            uint8_t* slot = stream.wait(j, 0);
            v.check(slot == ring.p + (size_t) (j % kSlots) * slot_bytes, "job j uses slot j % slots");
            if (j % 7 == 0) spin<<<1, 1>>>(2000000);   // about 1 ms: readers must not overwrite the slot meanwhile
            ck(cudaMemcpyAsync(check.p + (size_t) j * slot_bytes, slot, slot_bytes, cudaMemcpyDeviceToDevice, 0), "copy out");
            stream.release(j, 0);
        }
        stream.drain();
        const auto got = check.down();
        int bad = 0;
        for (int j = 0; j < kJobs; ++j) {
            const auto& x = pack.expert(jobs[j].first, jobs[j].second);
            bad += std::memcmp(got.data() + (size_t) j * slot_bytes, pack.expert_base() + x.offset, slot_bytes) != 0;
        }
        const auto st = stream.take_stats();
        std::printf("round %d: %d of %d slots wrong; jobs %lld (cache %lld, ssd %lld, ram %lld), consumer waited %.1f ms\n",
                    round, bad, kJobs, (long long) st.jobs, (long long) st.from_cache, (long long) st.from_ssd,
                    (long long) st.from_ram, st.consumer_wait_ms);
        v.check(bad == 0, "a slot held other bytes than its job's expert");
        v.check(st.jobs == kJobs, "every job counted once");
    }
    }
    return v.finish();
}
