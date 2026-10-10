// src/ds41/tests/engine_reset_test.cpp - what ds41_serve needs from the engine between requests.
//   --case reset      reset() starts over at position 0: a prompt after reset gives the logits it gave before (after
//                     a prefill, after decode steps, after a pending verify window); token by token, those of the
//                     fresh engine
//   --case progress   the prefill callback runs at least once per layer, with done rising to the total
//   --case cancel     a callback that returns false stops the prefill inside its pass: PrefillCancelled, position 0,
//                     and the next prompt gives the reference logits
//   --case snapshot   restore_snapshot() goes back to a saved position: after decode steps past it, and after a
//                     pending verify window, the next tokens give the logits of a fresh engine fed the same tokens;
//                     a snapshot past the current position, or an empty slot, is refused
// Needs the model pack: --pack DIR (without it the test is skipped with code 77).
#include "strata/ds41/config.hpp"
#include "strata/ds41/engine.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace strata::ds41;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

std::vector<int> prompt(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(3, 127999);   // ordinary tokens, not the specials
    std::vector<int> v = {0};                           // BOS
    while ((int) v.size() < n) v.push_back(d(rng));
    return v;
}

bool throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

/// The largest |a - b| over the logits, and whether the argmax agrees
bool same_logits(const std::vector<float>& a, const std::vector<float>& b, const std::string& what) {
    double worst = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) worst = std::max(worst, (double) std::fabs(a[i] - b[i]));
    const auto am = std::max_element(a.begin(), a.end()) - a.begin();
    const auto bm = std::max_element(b.begin(), b.end()) - b.begin();
    std::printf("  %s: max |diff| %.6f, argmax %ld vs %ld\n", what.c_str(), worst, (long) am, (long) bm);
    return a.size() == b.size() && am == bm && worst < 1e-2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string pack, which = "reset", profile;
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
    opt.max_seq = 4096;
    opt.cpu_threads = 8;
    opt.expert_profile = profile;
    opt.vram_expert_slots = profile.empty() ? 0 : -1;
    opt.adapt_every = 0;   // static residency: the same experts on the same path in every run
    // the RAM tier static too (adaptive is the default): its prefetch computes guessed RAM experts on the GPU, which
    // follows the tier's history
    setenv("DS41_RAM_ADAPT", "0", 1);
    const std::vector<int> a = prompt(700, 1), b = prompt(600, 2);
    try {
        Engine e(pack, opt);
        check(e.position() == 0, "a new engine is at position 0");
        // token by token on the fresh engine: the reference of the step path (prefill's logits differ slightly)
        const std::vector<int> b64(b.begin(), b.begin() + 64);
        std::vector<float> ref_steps;
        if (which == "reset") {
            for (int i = 0; i < (int) b64.size(); ++i) e.step(b64[i], i);
            ref_steps = e.last_logits();
            e.reset();
        }
        e.prefill(b, 0);
        const std::vector<float> ref = e.last_logits();
        check(e.position() == (int) b.size(), "position() counts the prompt");
        if (which == "reset") {
            e.reset();
            check(e.position() == 0, "reset() goes back to position 0");
            e.prefill(a, 0);
            int next = e.step(5, (int) a.size());
            for (int i = 1; i < 8; ++i) next = e.step(next, (int) a.size() + i);
            e.reset();
            e.prefill(b, 0);
            check(same_logits(e.last_logits(), ref, "after a prompt and decode steps"), "reset after decode");
            e.verify({7, 8, 9}, (int) b.size());   // a pending window, then reset instead of commit
            e.reset();
            e.prefill(b, 0);
            check(same_logits(e.last_logits(), ref, "after a pending verify window"), "reset drops a pending window");
            e.reset();
            for (int i = 0; i < (int) b64.size(); ++i) e.step(b64[i], i);
            check(same_logits(e.last_logits(), ref_steps, "token by token after reset"), "step() after reset");
        } else if (which == "progress") {
            e.reset();
            std::vector<std::pair<int, int>> calls;
            e.set_prefill_progress([&](int done, int total) {
                calls.push_back({done, total});
                return true;
            });
            e.prefill(a, 0);
            bool rising = true;
            for (size_t i = 1; i < calls.size(); ++i) rising = rising && calls[i].first >= calls[i - 1].first;
            std::printf("  %zu calls, last %d of %d\n", calls.size(), calls.empty() ? -1 : calls.back().first,
                        calls.empty() ? -1 : calls.back().second);
            check(calls.size() >= (size_t) kLayers, "at least one call per layer");
            check(rising, "done never goes down");
            check(!calls.empty() && calls.back().first == (int) a.size() && calls.back().second == (int) a.size(),
                  "the last call reports all tokens done");
        } else if (which == "cancel") {
            e.reset();
            int calls = 0;
            e.set_prefill_progress([&](int done, int total) {
                ++calls;
                return done < total / 2;   // stop in the middle of the pass
            });
            bool cancelled = false;
            try {
                e.prefill(a, 0);
            } catch (const PrefillCancelled&) {
                cancelled = true;
            }
            check(cancelled, "a false from the callback throws PrefillCancelled");
            check(calls > 1 && calls < kLayers, "it stops inside the pass (" + std::to_string(calls) + " calls)");
            check(e.position() == 0, "the engine is back at position 0");
            e.set_prefill_progress(nullptr);
            e.prefill(b, 0);
            check(same_logits(e.last_logits(), ref, "after a cancelled prefill"), "the next prompt is right");
        } else if (which == "snapshot") {
            // reference: a fresh start fed a, then b's first 40 tokens (a batched prefill, as below)
            const std::vector<int> tail(b.begin(), b.begin() + 40);
            e.reset();
            e.prefill(a, 0);
            e.prefill(tail, (int) a.size());
            const std::vector<float> want = e.last_logits();
            // the case: a, a snapshot, decode steps past it, back to the snapshot, then the same tail
            e.reset();
            check(e.snapshot_slots() >= 2, "the engine has snapshot slots");
            e.prefill(a, 0);
            e.save_snapshot(1);
            int next = e.step(5, (int) a.size());
            for (int i = 1; i < 12; ++i) next = e.step(next, (int) a.size() + i);
            check(e.restore_snapshot(1) == (int) a.size() && e.position() == (int) a.size(),
                  "restore goes back to the saved position");
            e.prefill(tail, (int) a.size());
            check(same_logits(e.last_logits(), want, "after 12 decode steps and a restore"), "restore after decode");
            e.restore_snapshot(1);
            e.verify({7, 8, 9}, (int) a.size());   // a pending window, then a restore instead of commit
            e.restore_snapshot(1);
            e.prefill(tail, (int) a.size());
            check(same_logits(e.last_logits(), want, "after a pending verify window"), "restore drops a pending window");
            // the snapshot stays: a second restore works the same way (token by token this time)
            e.restore_snapshot(1);
            for (int i = 0; i < (int) tail.size(); ++i) e.step(tail[i], (int) a.size() + i);
            const std::vector<float> by_steps = e.last_logits();
            e.restore_snapshot(1);
            for (int i = 0; i < (int) tail.size(); ++i) e.step(tail[i], (int) a.size() + i);
            check(same_logits(e.last_logits(), by_steps, "token by token, twice"), "a snapshot can be restored again");
            e.reset();
            e.prefill(b, 0);   // shorter than the snapshot's position
            check(throws([&] { e.restore_snapshot(1); }), "a snapshot past the current position is refused");
            check(throws([&] { e.restore_snapshot(0); }), "an empty slot is refused");
            check(throws([&] { e.save_snapshot(e.snapshot_slots()); }), "a slot past the last is refused");
        } else {
            std::printf("unknown --case %s\n", which.c_str());
            return 2;
        }
    } catch (const std::exception& ex) {
        std::printf("FAIL: unexpected exception: %s\n", ex.what());
        ++failures;
    }
    std::printf("RESULT %s engine_reset %s\n", failures ? "fail" : "pass", which.c_str());
    return failures ? 1 : 0;
}
