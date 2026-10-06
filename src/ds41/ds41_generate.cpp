// src/ds41/ds41_generate.cpp - run the M1 engine on token ids: greedy generation, timings, optional dump.
//
//   ds41_generate --pack DIR --ids 0,128000,... [--gen 32] [--threads 8] [--dump steps.bin] [--force-ids FILE]
//                 [--expert-profile ds41/data/expert-profile.bin [--vram-slots N] [--adapt-every 4] [--adapt-swaps 96]
//                  [--ram-budget-gib N (0 none: default, -1 available RAM less 4 GB)]]
//                 [--prefill [--prefill-chunk 65536] [--prefill-batch 4096] [--prefill-ring 256] [--prefill-threads 8]]
//
// --prefill runs the prompt (or, with --force-ids, the whole forced sequence for its nll) through the batched
// prefill (M3) instead of step() token by token; generation then continues with step().
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
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::ds41;

/// comma-separated ids, or @FILE holding them (a 32K-token prompt is longer than one command-line argument may be)
static std::vector<int> parse_ids(const std::string& arg) {
    std::string s = arg;
    if (!s.empty() && s[0] == '@') {
        std::ifstream f(s.substr(1));
        if (!f) throw std::runtime_error("cannot open " + s.substr(1));
        s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoi(tok));
    return v;
}

int main(int argc, char** argv) {
    std::string pack, ids_s, dump_path, force_path;
    int gen = 32;
    bool batched = false;
    EngineOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--pack") pack = next();
        else if (a == "--ids") ids_s = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--threads") opt.cpu_threads = std::stoi(next());
        else if (a == "--max-seq") opt.max_seq = std::stoi(next());
        else if (a == "--expert-profile") opt.expert_profile = next();
        else if (a == "--vram-slots") opt.vram_expert_slots = std::stoll(next());
        else if (a == "--adapt-every") opt.adapt_every = std::stoi(next());
        else if (a == "--adapt-swaps") opt.adapt_swaps = std::stoi(next());
        else if (a == "--ram-budget-gib") opt.ram_budget_gib = std::stod(next());
        else if (a == "--prefill") batched = true;
        else if (a == "--prefill-chunk") opt.prefill_chunk = std::stoi(next());
        else if (a == "--prefill-ring") opt.prefill_ring = std::stoi(next());
        else if (a == "--prefill-batch") opt.prefill_batch = std::stoi(next());
        else if (a == "--prefill-threads") opt.prefill_threads = std::stoi(next());
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
        Engine engine(pack, opt);
        if (batched) {   // prefill the prompt (or the forced sequence) in one call, then decode with step()
            const std::vector<int>& pre = forced.empty() ? prompt : forced;
            std::vector<float> nll;
            int next = engine.prefill(pre, 0, forced.empty() ? nullptr : &nll);
            const auto& p = engine.last_prefill();
            std::printf("prefill_tokens %zu ms %.1f tok_s %.1f chunks %d chunk_tokens %d sub_batch %d engram_ms %.1f stream_wait_ms %.1f"
                        " vram_experts %lld streamed %lld (ram %lld cache %lld ssd %lld) engram_rows %lld unique %lld\n",
                        pre.size(), p.total_ms, pre.size() / (p.total_ms / 1000.0), p.chunks, p.chunk_tokens, p.sub_batch,
                        p.engram_ms,
                        p.stream_wait_ms, (long long) p.vram_experts, (long long) p.streamed, (long long) p.from_ram,
                        (long long) p.from_cache, (long long) p.from_ssd, (long long) p.engram_rows,
                        (long long) p.engram_unique);
            if (!nll.empty()) {
                double sum = 0;
                for (float v : nll) sum += v;
                std::printf("teacher_forced_mean_nll %.6f ppl %.4f over %zu tokens\n", sum / nll.size(),
                            std::exp(sum / nll.size()), nll.size());
            }
            if (forced.empty()) {
                std::vector<int> out = {next};
                double ms = 0;
                for (int i = 1; i < gen; ++i) {
                    next = engine.step(next, (int) prompt.size() + i - 1);
                    ms += engine.last_timing().total_ms;
                    out.push_back(next);
                }
                std::printf("generated:");
                for (int v : out) std::printf(" %d", v);
                std::printf("\ndecode_ms_per_token %.1f over %d steps\n", gen > 1 ? ms / (gen - 1) : 0.0, gen - 1);
            }
            return 0;
        }
        std::FILE* dump = dump_path.empty() ? nullptr : std::fopen(dump_path.c_str(), "wb");
        StepDump sd;
        const int total = forced.empty() ? (int) prompt.size() + gen : (int) forced.size();
        int next = -1;
        std::vector<int> out;
        double decode_ms = 0, nll_sum = 0, cpu_ms = 0;
        long long hits = 0, routed = 0, ram = 0, file = 0, ssd = 0, warmed = 0, useful = 0;
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
            if (pos >= (int) prompt.size() - 1) { decode_ms += t.total_ms; cpu_ms += t.cpu_experts_ms; ++decode_steps; }
            hits += t.expert_hits;
            ram += t.ram_experts;
            file += t.file_experts;
            ssd += t.ssd_experts;
            warmed += t.warmed;
            useful += t.warmed_useful;
            routed += t.expert_total;
            if (pos >= (int) prompt.size() - 1 && forced.empty()) out.push_back(next);
            std::fprintf(stderr,
                         "pos %d tok %d -> %d  total %.1f ms (engram reads %.1f, layers %.1f, of which cpu experts %.1f)"
                         "  vram hits %d/%d swaps %d  cpu: ram %d file %d (ssd %d)  warmed %d useful %d\n",
                         pos, tok, next, t.total_ms, t.engram_ms, t.gpu_ms, t.cpu_experts_ms, t.expert_hits,
                         t.expert_total, t.vram_swaps, t.ram_experts, t.file_experts, t.ssd_experts, t.warmed,
                         t.warmed_useful);
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
        std::printf("\ndecode_ms_per_token %.1f over %d steps (cpu experts %.1f)\n",
                    decode_steps ? decode_ms / decode_steps : 0.0, decode_steps,
                    decode_steps ? cpu_ms / decode_steps : 0.0);
        std::printf("vram_expert_slots %d hit_rate %.4f\n", engine.vram_expert_slots(),
                    routed ? (double) hits / routed : 0.0);
        std::printf("tiers_share vram %.4f ram %.4f file %.4f ssd %.4f\n", routed ? (double) hits / routed : 0.0,
                    routed ? (double) ram / routed : 0.0, routed ? (double) file / routed : 0.0,
                    routed ? (double) ssd / routed : 0.0);
        std::printf("lookahead warmed %lld useful %lld (%.3f)\n", warmed, useful, warmed ? (double) useful / warmed : 0.0);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
