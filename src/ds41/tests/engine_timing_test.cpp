// src/ds41/tests/engine_timing_test.cpp - the decode step's critical-path timing (Engine::Timing).
//   The CPU expert worker reports where its part of the step went: waiting for the GPU to publish a layer, reading
//   experts into the adaptive RAM tier, computing experts. The parts must fit inside the step, so the decode profile
//   (ds41_generate --step-log) can split a step's time without a GPU trace.
// Needs the model pack and the expert profile: --pack DIR --expert-profile FILE (without them the test is skipped
// with code 77).
#include "strata/ds41/config.hpp"
#include "strata/ds41/engine.hpp"

#include <cstdio>
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
    std::string pack, profile;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--pack") pack = argv[i + 1];
        else if (a == "--expert-profile") profile = argv[i + 1];
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
    const std::vector<int> prompt = {0, 128000, 1234, 5678, 42, 4096, 777, 31337};
    const double eps = 0.5;   // ms: the parts are timed by separate clock reads
    try {
        Engine e(pack, opt);
        int next = 0;
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
            check(t.worker_wait_ms + t.admit_ms + t.cpu_experts_ms <= t.worker_span_ms + eps,
                  "wait + reads + compute fit in the worker's span" + at);
            check(t.swaps_ms >= 0 && t.end_ms >= 0 && t.swaps_ms + t.end_ms + t.engram_ms <= t.total_ms + eps,
                  "the swap and end-of-step bookkeeping fit in the step" + at);
        }
        check(next >= 0 && next < kVocab, "the last step returns a token");
    } catch (const std::exception& ex) {
        std::printf("FAIL: exception %s\n", ex.what());
        ++failures;
    }
    std::printf("RESULT %s engine_timing\n", failures ? "fail" : "pass");
    return failures ? 1 : 0;
}
