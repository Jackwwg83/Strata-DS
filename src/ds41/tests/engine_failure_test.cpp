// src/ds41/tests/engine_failure_test.cpp - the engine's input checks and failure paths (PR #10 review).
//   --case tokens       an out-of-range token is refused before any state changes; the engine stays usable
//   --case engram       a failure before the step reaches the device: the history is restored, the step can be retried
//   --case worker       an exception in the CPU expert worker is rethrown by the step (not std::terminate); the engine
//                       then refuses every call
//   --case prefill      a failure before the first prefill pass leaves the engine usable; one inside a pass does not
//   --case lifecycle    two engines one after the other: the second does not run out of VRAM, and the free VRAM
//                       comes back after each destruction
// Needs the model pack: --pack DIR (without it the test is skipped with code 77). DS41_TEST_FAULT arms the faults.
#include "strata/ds41/config.hpp"
#include "strata/ds41/engine.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::ds41;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
bool throws(const std::function<void()>& fn, std::string* message = nullptr) {
    try {
        fn();
    } catch (const std::exception& e) {
        if (message) *message = e.what();
        return true;
    }
    return false;
}
size_t free_vram() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    std::string pack, which = "tokens", profile;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--pack") pack = argv[i + 1];
        else if (a == "--case") which = argv[i + 1];
        else if (a == "--expert-profile") profile = argv[i + 1];
    }
    if (pack.empty()) {
        std::printf("RESULT skip (no --pack)\n");
        return 77;
    }
    EngineOptions opt;
    opt.max_seq = 1024;
    opt.cpu_threads = 8;
    opt.expert_profile = profile;
    opt.vram_expert_slots = profile.empty() ? 0 : 64;
    opt.ram_budget_gib = 0;
    const std::vector<int> prompt = {0, 128000, 1234, 5678, 42, 4096, 777, 31337};
    try {
        if (which == "tokens") {
            Engine e(pack, opt);
            check(throws([&] { e.step(-1, 0); }), "step(-1) is refused");
            check(throws([&] { e.step(kVocab, 0); }), "step(kVocab) is refused");
            check(throws([&] { e.prefill({1, kVocab + 5, 2}, 0); }), "prefill with an out-of-range token is refused");
            int next = -1;
            check(!throws([&] { next = e.step(prompt[0], 0); }), "the engine still steps at position 0 afterwards");
            check(next >= 0 && next < kVocab, "and returns a token");
        } else if (which == "engram") {
            setenv("DS41_TEST_FAULT", "engram", 1);
            Engine e(pack, opt);
            std::string msg;
            check(throws([&] { e.step(prompt[0], 0); }, &msg) && msg.find("engram") != std::string::npos,
                  "the injected engram failure surfaces");
            int next = -1;
            check(!throws([&] { next = e.step(prompt[0], 0); }), "the same position can be retried");
            check(!throws([&] { e.step(prompt[1], 1); }), "and decoding continues");
        } else if (which == "worker") {
            setenv("DS41_TEST_FAULT", "worker", 1);
            Engine e(pack, opt);
            std::string msg;
            check(throws([&] { e.step(prompt[0], 0); }, &msg) && msg.find("worker") != std::string::npos,
                  "the worker's exception is rethrown by the step (the process survives)");
            check(throws([&] { e.step(prompt[1], 1); }, &msg) && msg.find("unusable") != std::string::npos,
                  "the engine refuses the next step");
            check(throws([&] { e.prefill(prompt, 1); }), "and prefill");
        } else if (which == "prefill") {
            {
                setenv("DS41_TEST_FAULT", "prefill", 1);
                Engine e(pack, opt);
                check(throws([&] { e.prefill(prompt, 0); }), "a failure before the first pass surfaces");
                check(!throws([&] { e.prefill(prompt, 0); }), "the engine is still usable: prefill again from 0");
            }
            {
                setenv("DS41_TEST_FAULT", "prefill_pass", 1);
                Engine e(pack, opt);
                check(throws([&] { e.prefill(prompt, 0); }), "a failure inside a pass surfaces");
                std::string msg;
                check(throws([&] { e.step(prompt[0], 0); }, &msg) && msg.find("unusable") != std::string::npos,
                      "the engine refuses further calls");
            }
        } else if (which == "lifecycle") {
            const size_t f0 = free_vram();
            for (int round = 0; round < 2; ++round) {
                {
                    Engine e(pack, opt);
                    e.prefill(prompt, 0);
                }
                cudaDeviceSynchronize();
                const size_t f = free_vram();
                const double lost = ((double) f0 - (double) f) / (1 << 20);
                std::printf("round %d: free VRAM %.0f MiB less than before the first engine\n", round, lost);
                check(lost < 256.0, "VRAM comes back after the engine is destroyed (round " + std::to_string(round) + ")");
            }
        } else {
            std::printf("unknown --case %s\n", which.c_str());
            return 2;
        }
    } catch (const std::exception& e) {
        std::printf("FAIL: unexpected exception: %s\n", e.what());
        ++failures;
    }
    std::printf("RESULT %s engine_failure %s\n", failures ? "fail" : "pass", which.c_str());
    return failures ? 1 : 0;
}
