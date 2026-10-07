// src/ds41/tests/batch_bench.cpp - throughput of the batch slots: N real chats (tools/ds41/chat_ids.py) read into N
// slots, then G steps of all of them together. Prints the mean step (after 8 warm-up steps) and the tokens per second
// of all rows together, with where a step's time went.
//   batch_bench --pack DIR --expert-profile F --ids-dir D --slots N [--gen G] [--threads T]
// N = 1 decodes one chat with step() (the single-request path) for comparison.
#include "strata/ds41/config.hpp"
#include "strata/ds41/engine.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace strata::ds41;

int main(int argc, char** argv) {
    std::string pack, profile, ids_dir;
    int slots = 2, gen = 96, threads = 16;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--pack") pack = argv[i + 1];
        else if (a == "--expert-profile") profile = argv[i + 1];
        else if (a == "--ids-dir") ids_dir = argv[i + 1];
        else if (a == "--slots") slots = std::stoi(argv[i + 1]);
        else if (a == "--gen") gen = std::stoi(argv[i + 1]);
        else if (a == "--threads") threads = std::stoi(argv[i + 1]);
    }
    if (pack.empty() || ids_dir.empty()) {
        std::printf("usage: batch_bench --pack DIR --expert-profile F --ids-dir D --slots N [--gen G]\n");
        return 2;
    }
    std::vector<std::vector<int>> prompts;
    for (const char* name : {"code", "zh_chat", "en_explain", "agent"}) {
        std::FILE* f = std::fopen((ids_dir + "/" + name + ".ids").c_str(), "r");
        if (!f) continue;
        std::vector<int> ids;
        int v = 0;
        while (std::fscanf(f, "%d,", &v) == 1) ids.push_back(v);
        std::fclose(f);
        prompts.push_back(ids);
    }
    EngineOptions opt;
    opt.max_seq = 8192;
    opt.cpu_threads = threads;
    opt.expert_profile = profile;
    opt.batch_slots = slots > 1 ? slots : 0;
    Engine e(pack, opt);
    const int n = slots > 1 ? slots : 1;
    std::vector<int> x(n), rows(n);
    for (int k = 0; k < n; ++k) {
        e.reset();
        x[k] = e.prefill(prompts[k % prompts.size()], 0);
        rows[k] = k;
        if (slots > 1) e.copy_to_slot(k);
    }
    double ms = 0, gpu = 0, engram = 0, cpu = 0;
    int64_t hits = 0, total = 0, counted = 0;
    for (int i = 0; i < gen; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (slots > 1) x = e.step_slots(rows, x);
        else x[0] = e.step(x[0], e.position());
        const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (i < 8) continue;
        const auto& tm = e.last_timing();
        ms += dt;
        gpu += tm.gpu_ms;
        engram += tm.engram_ms;
        cpu += tm.cpu_experts_ms;
        hits += tm.expert_hits;
        total += tm.expert_total;
        ++counted;
    }
    const double step = ms / counted;
    std::printf("slots %d: %.1f ms per step, %.1f tokens/s in all, %.1f per row; gpu %.1f engram %.1f cpu %.1f, "
                "VRAM hits %.2f\n",
                n, step, 1000.0 * n / step, 1000.0 / step, gpu / counted, engram / counted, cpu / counted,
                total ? (double) hits / total : 0.0);
    return 0;
}
