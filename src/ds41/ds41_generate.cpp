// src/ds41/ds41_generate.cpp - run the M1 engine on token ids: greedy generation, timings, optional dump.
//
//   ds41_generate --pack DIR --ids 0,128000,... [--gen 32] [--threads 8] [--dump steps.bin] [--force-ids FILE]
//
// --force-ids feeds a fixed token sequence (from the oracle) instead of the engine's own predictions, so a
// per-layer comparison stays aligned even after the first differing token. Tokenization stays in Python.
#include "strata/ds41/config.hpp"
#include "strata/ds41/engine.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::ds41;

static std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoi(tok));
    return v;
}

int main(int argc, char** argv) {
    std::string pack, ids_s, dump_path, force_path;
    int gen = 32, threads = 8, max_seq = 4096;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--pack") pack = next();
        else if (a == "--ids") ids_s = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--max-seq") max_seq = std::stoi(next());
        else if (a == "--dump") dump_path = next();
        else if (a == "--force-ids") force_path = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (pack.empty() || (ids_s.empty() && force_path.empty())) {
        std::fprintf(stderr, "usage: ds41_generate --pack DIR --ids a,b,c [--gen N] [--threads T] [--dump F] [--force-ids F]\n");
        return 2;
    }
    std::vector<int> prompt = parse_ids(ids_s), forced;
    if (!force_path.empty()) {
        std::ifstream f(force_path);
        std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        forced = parse_ids(all);
    }
    try {
        Engine engine(pack, max_seq, threads);
        std::FILE* dump = dump_path.empty() ? nullptr : std::fopen(dump_path.c_str(), "wb");
        StepDump sd;
        const int total = forced.empty() ? (int) prompt.size() + gen : (int) forced.size();
        int next = -1;
        std::vector<int> out;
        double decode_ms = 0, nll_sum = 0;
        int decode_steps = 0, nll_n = 0;
        for (int pos = 0; pos < total; ++pos) {
            const int tok = !forced.empty() ? forced[pos] : pos < (int) prompt.size() ? prompt[pos] : next;
            next = engine.step(tok, pos, dump ? &sd : nullptr);
            const auto& t = engine.last_timing();
            if (!forced.empty() && pos + 1 < (int) forced.size()) {    // teacher-forced -log p(next token)
                const auto& lg = engine.last_logits();
                double mx = -1e300, se = 0;
                for (float v : lg) mx = std::max(mx, (double) v);
                for (float v : lg) se += std::exp((double) v - mx);
                nll_sum += mx + std::log(se) - lg[forced[pos + 1]];
                ++nll_n;
            }
            if (pos >= (int) prompt.size() - 1) { decode_ms += t.total_ms; ++decode_steps; }
            if (pos >= (int) prompt.size() - 1 && forced.empty()) out.push_back(next);
            std::fprintf(stderr, "pos %d tok %d -> %d  total %.1f ms (cpu experts %.1f, engram %.1f, gpu+sync %.1f)\n",
                         pos, tok, next, t.total_ms, t.cpu_experts_ms, t.engram_ms, t.gpu_ms);
            if (dump) {
                const int32_t hdr[2] = {tok, next};
                std::fwrite(hdr, 4, 2, dump);
                std::fwrite(sd.hidden.data(), 2, sd.hidden.size(), dump);
                for (int l = 0; l < kLayers; ++l) std::fwrite(sd.routes[l].data(), 4, kTopK, dump);
                for (int l = 0; l < kLayers; ++l) std::fwrite(sd.weights[l].data(), 4, kTopK, dump);
                for (auto& p : sd.top_logits) { std::fwrite(&p.first, 4, 1, dump); }
                for (auto& p : sd.top_logits) { std::fwrite(&p.second, 4, 1, dump); }
            }
        }
        if (dump) std::fclose(dump);
        std::printf("generated:");
        for (int v : out) std::printf(" %d", v);
        if (nll_n) std::printf("\nteacher_forced_mean_nll %.6f ppl %.4f over %d tokens", nll_sum / nll_n,
                               std::exp(nll_sum / nll_n), nll_n);
        std::printf("\ndecode_ms_per_token %.1f over %d steps\n", decode_steps ? decode_ms / decode_steps : 0.0,
                    decode_steps);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
