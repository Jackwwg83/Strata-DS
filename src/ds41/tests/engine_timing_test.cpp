// src/ds41/tests/engine_timing_test.cpp - the decode step's critical-path timing (Engine::Timing).
//   The CPU expert worker reports where its part of the step went: waiting for the GPU to publish a layer, reading
//   experts into the adaptive RAM tier, computing experts. The parts must fit inside the step, so the decode profile
//   (ds41_generate --step-log) can split a step's time without a GPU trace.
//   --case plain      no RAM tier: every CPU expert comes from the file
//   --case adaptive   a 2 GiB adaptive RAM tier (DS41_RAM_ADAPT=8): misses are read into free slots, so the reads
//                     (admit_ms) are timed too
// Needs the model pack and the expert profile: --pack DIR --expert-profile FILE (without them the test is skipped
// with code 77).
#include "strata/ds41/config.hpp"
#include "strata/ds41/engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::ds41;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
    std::string pack, profile, which = "plain";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--pack") pack = argv[i + 1];
        else if (a == "--expert-profile") profile = argv[i + 1];
        else if (a == "--case") which = argv[i + 1];
    }
    if (which != "plain" && which != "adaptive") {
        std::printf("RESULT fail (unknown --case %s)\n", which.c_str());
        return 1;
    }
    if (pack.empty() || profile.empty()) {
        std::printf("RESULT skip (no --pack or --expert-profile)\n");
        return 77;
    }
    EngineOptions opt;
    opt.max_seq = 1024;
    opt.cpu_threads = 8;
    opt.expert_profile = profile;
    opt.vram_expert_slots = 64;   // most routed experts miss VRAM: the CPU worker runs every layer
    opt.ram_budget_gib = 0;
    if (which == "adaptive") {
        opt.ram_budget_gib = 2;
        setenv("DS41_RAM_ADAPT", "8", 1);
    }
    const std::vector<int> prompt = {0, 128000, 1234, 5678, 42, 4096, 777, 31337};
    const double eps = 0.5;   // ms: the parts are timed by separate clock reads
    try {
        Engine e(pack, opt);
        int next = 0, reads = 0;
        double admit = 0;
        for (size_t p = 0; p < prompt.size(); ++p) {
            next = e.step(prompt[p], int(p));
            const auto& t = e.last_timing();
            const std::string at = " (step " + std::to_string(p) + ")";
            check(t.cpu_experts() > 0, "the CPU worker computed experts" + at);
            check(t.worker_span_ms > 0 && t.worker_span_ms <= t.total_ms + eps,
                  "the worker's span is inside the step" + at);
            check(t.worker_lead_ms >= 0 && t.worker_lead_ms + t.worker_span_ms <= t.total_ms + eps,
                  "the lead before the worker wakes and its span fit in the step" + at);
            check(t.worker_wait_ms > 0 && t.worker_first_wait_ms > 0 && t.worker_first_wait_ms <= t.worker_wait_ms + eps,
                  "the worker waits for the GPU, layer 0 included" + at);
            check(t.admit_ms >= 0, "the RAM tier read time is not negative" + at);
            if (which == "plain") check(t.admit_ms == 0, "no RAM tier: no reads into it" + at);
            admit += t.admit_ms;
            reads += t.ssd_experts;
            check(t.worker_wait_ms + t.admit_ms + t.cpu_experts_ms <= t.worker_span_ms + eps,
                  "wait + reads + compute fit in the worker's span" + at);
            check(t.swaps_ms >= 0 && t.end_ms >= 0 && t.swaps_ms + t.end_ms + t.engram_ms <= t.total_ms + eps,
                  "the swap and end-of-step bookkeeping fit in the step" + at);
        }
        check(next >= 0 && next < kVocab, "the last step returns a token");
        if (which == "adaptive")
            check(reads > 0 && admit > 0, "the adaptive tier read experts from the SSD (" + std::to_string(reads) +
                                              ") and timed it (" + std::to_string(admit) + " ms)");
    } catch (const std::exception& ex) {
        std::printf("FAIL: exception %s\n", ex.what());
        ++failures;
    }
    std::printf("RESULT %s engine_timing %s\n", failures ? "fail" : "pass", which.c_str());
    return failures ? 1 : 0;
}
