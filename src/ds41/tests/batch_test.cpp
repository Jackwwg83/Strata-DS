// src/ds41/tests/batch_test.cpp - batch slots: three conversations decoded together give, row by row, the tokens each
// gives decoded alone (static residency, so the same experts take the same path); a step of two of the slots in
// another row order gives the same; a slot copied back to the main session continues there as it would have alone.
// Needs the model pack: --pack DIR [--expert-profile F] [--ids-dir D: real chats instead of random tokens] (without a
// pack the test is skipped with code 77).
#include "strata/ds41/config.hpp"
#include "strata/ds41/engine.hpp"

#include <cstdlib>
#include <cstdio>
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
    std::uniform_int_distribution<int> d(3, 127999);
    std::vector<int> v = {0};
    while ((int) v.size() < n) v.push_back(d(rng));
    return v;
}
std::string show(const std::vector<int>& v) {
    std::string s;
    for (int t : v) s += std::to_string(t) + " ";
    return s;
}
}  // namespace

int main(int argc, char** argv) {
    std::string pack, profile, ids_dir;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--pack") pack = argv[i + 1];
        else if (a == "--expert-profile") profile = argv[i + 1];
        else if (a == "--ids-dir") ids_dir = argv[i + 1];   // tools/ds41/chat_ids.py: real chats (code, zh_chat, en_explain)
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
    opt.adapt_every = 0;   // static residency
    // (adapt_every 0 keeps the RAM tier static too: the engine's rule, not the test's)
    opt.batch_slots = 3;
    constexpr int kGen = 20, kBatched = 14;
    std::vector<std::vector<int>> prompts = {prompt(300, 1), prompt(41, 2), prompt(150, 3)};
    if (!ids_dir.empty()) {
        prompts.clear();
        for (const char* name : {"code", "zh_chat", "en_explain"}) {
            std::FILE* f = std::fopen((ids_dir + "/" + name + ".ids").c_str(), "r");
            if (!f) {
                std::printf("FAIL: cannot read %s/%s.ids\n", ids_dir.c_str(), name);
                return 1;
            }
            std::vector<int> ids;
            int v = 0;
            while (std::fscanf(f, "%d,", &v) == 1) ids.push_back(v);
            std::fclose(f);
            prompts.push_back(ids);
        }
    }
    try {
        Engine e(pack, opt);
        check(e.batch_slots() == 3, "the engine has 3 batch slots");
        // alone: prompt, then kGen greedy tokens (the first from the prompt's logits)
        std::vector<std::vector<int>> alone(3);
        for (int k = 0; k < 3; ++k) {
            e.reset();
            int x = e.prefill(prompts[k], 0);
            for (int i = 0; i < kGen; ++i) {
                alone[k].push_back(x);
                x = e.step(x, (int) prompts[k].size() + i);
            }
            std::printf("  alone %d: %s\n", k, show(alone[k]).c_str());
        }
        // together: each prompt read in the main session, copied into slot k; then kBatched steps of all slots
        std::vector<std::vector<int>> together(3);
        std::vector<int> x(3);
        for (int k = 0; k < 3; ++k) {
            e.reset();
            x[k] = e.prefill(prompts[k], 0);
            together[k].push_back(x[k]);
            e.copy_to_slot(k);
            check(e.slot_position(k) == (int) prompts[k].size(), "slot " + std::to_string(k) + " holds its prompt");
        }
        for (int i = 1; i < kBatched; ++i) {
            // every third step: two of the slots, in another order
            const bool two = i % 3 == 0;
            const std::vector<int> rows = two ? std::vector<int>{2, 0} : std::vector<int>{0, 1, 2};
            std::vector<int> toks;
            for (int r : rows) toks.push_back(x[r]);
            const std::vector<int> next = e.step_slots(rows, toks);
            for (size_t t = 0; t < rows.size(); ++t) {
                x[rows[t]] = next[t];
                together[rows[t]].push_back(next[t]);
            }
        }
        for (int k = 0; k < 3; ++k) {
            std::printf("  together %d: %s\n", k, show(together[k]).c_str());
            const std::vector<int> want(alone[k].begin(), alone[k].begin() + (long) together[k].size());
            check(together[k] == want, "slot " + std::to_string(k) + ": the tokens decoded alone");
        }
        // slot 1 back to the main session: it continues as alone
        const int n1 = (int) together[1].size();
        e.copy_from_slot(1);
        check(e.position() == (int) prompts[1].size() + n1 - 1, "the main session holds slot 1's tokens");
        int y = e.step(together[1].back(), e.position());
        check(n1 < kGen && y == alone[1][n1], "slot 1 continues in the main session as alone");
    } catch (const std::exception& ex) {
        std::printf("FAIL: unexpected exception: %s\n", ex.what());
        ++failures;
    }
    std::printf("RESULT %s batch\n", failures ? "fail" : "pass");
    return failures ? 1 : 0;
}
