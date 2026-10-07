// src/ds41/ds41_generate.cpp - run the M1 engine on token ids: greedy generation, timings, optional dump.
//
//   ds41_generate --pack DIR --ids 0,128000,... [--gen 32] [--threads 8] [--dump steps.bin] [--force-ids FILE]
//                 [--expert-profile ds41/data/expert-profile.bin [--vram-slots N] [--adapt-every 4] [--adapt-swaps 96]
//                  [--ram-budget-gib N (-1 available RAM less 24 GiB: default, as upstream setup; 0 none)]]
//                 [--prefill [--prefill-chunk 65536] [--prefill-batch 4096] [--prefill-ring 256] [--prefill-threads 8]]
//
// --prefill runs the prompt (or, with --force-ids, the whole forced sequence for its nll) through the batched
// prefill (M3) instead of step() token by token; generation then continues with step().
// --force-ids feeds a fixed token sequence (from the oracle) instead of the engine's own predictions, so a
// per-layer comparison stays aligned even after the first differing token. Tokenization stays in Python.
#include "strata/ds41/config.hpp"
#include "strata/ds41/engine.hpp"
#include "strata/ds41/suffix_drafter.hpp"

#include <algorithm>
#include <chrono>

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

/// The prefill line the speed scripts read (m3_context.sh): every field of PrefillTiming
static void print_prefill(const Engine& engine, size_t n) {
    const auto& p = engine.last_prefill();
    std::printf("prefill_tokens %zu ms %.1f tok_s %.1f chunks %d chunk_tokens %d sub_batch %d engram_ms %.1f stream_wait_ms %.1f"
                " vram_experts %lld streamed %lld (ram %lld cache %lld ssd %lld) engram_rows %lld unique %lld\n",
                n, p.total_ms, n / (std::max(p.total_ms, 1e-9) / 1000.0), p.chunks, p.chunk_tokens, p.sub_batch,
                p.engram_ms, p.stream_wait_ms, (long long) p.vram_experts, (long long) p.streamed,
                (long long) p.from_ram, (long long) p.from_cache, (long long) p.from_ssd, (long long) p.engram_rows,
                (long long) p.engram_unique);
}

// Output history contains raw IDs. It includes the pending target prediction.
static void generate(Engine& engine, const EngineOptions& opt, const std::vector<int>& prompt,
                     int count, bool batched, bool suffix, int max_t, int eos) {
    if (prompt.empty()) throw std::invalid_argument("generation needs a nonempty prompt");
    int next = -1;
    if (batched) {
        next = engine.prefill(prompt, 0);
        print_prefill(engine, prompt.size());
    } else {
        for (size_t p = 0; p < prompt.size(); ++p) next = engine.step(prompt[p], int(p));
    }
    SuffixDrafter drafter(3, 64, size_t(opt.max_seq)+8);
    for (int token : prompt) drafter.append(token);
    std::vector<int> out;
    if (count > 0) { out.push_back(next); drafter.append(next); }
    int pos = int(prompt.size()), rounds = 0, accepted = 0;
    double step_ms = 0;            // plain decode: the engine's step times, as the forced path reports them
    int64_t hits = 0, routed = 0;
    const auto begin = std::chrono::steady_clock::now();
    while (int(out.size()) < count && next != eos) {
        const int limit = std::min({max_t, count-int(out.size()), opt.max_seq-pos});
        if (limit < 1) throw std::runtime_error("generation exceeds max_seq");
        if (!suffix) {
            next = engine.step(next, pos++);
            const auto& tm = engine.last_timing();
            step_ms += tm.total_ms;
            hits += tm.expert_hits;
            routed += tm.expert_total;
            out.push_back(next); drafter.append(next); ++rounds;
            continue;
        }
        int32_t draft[8];
        const int proposed = drafter.propose(limit-1, draft);
        std::vector<int> window{next};
        window.insert(window.end(), draft, draft+proposed);
        auto result = engine.verify(window, pos);
        int keep = accepted_inputs(window, result.next);
        for (int i = 0; i < keep; ++i)
            if (result.next[i] == eos) { keep = i+1; break; }
        engine.commit(keep);
        std::printf("spec_window pos %d T %zu accepted %d emitted %d\n", pos, window.size(), keep-1, keep);
        for (int i = 0; i < keep; ++i) { out.push_back(result.next[i]); drafter.append(result.next[i]); }
        accepted += keep-1; ++rounds;
        next = result.next[keep-1]; pos += keep;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    const int decoded = std::max(0, int(out.size())-1);
    std::printf("generated:");
    for (int token : out) std::printf(" %d", token);
    std::printf("\ndecode_tokens %d windows %d accepted_drafts %d seconds %.6f tok_s %.3f\n",
                decoded, rounds, accepted, seconds, decoded/std::max(seconds, 1e-9));
    // the lines the speed scripts read (m3_context.sh); speculation: wall time per emitted token
    const double ms = suffix ? seconds * 1000.0 : step_ms;
    std::printf("decode_ms_per_token %.1f over %d steps\n", decoded > 0 ? ms / decoded : 0.0, decoded);
    if (!suffix)
        std::printf("vram_expert_slots %d hit_rate %.4f\n", engine.vram_expert_slots(),
                    routed ? (double) hits / routed : 0.0);
}

int main(int argc, char** argv) {
    std::string pack, ids_s, dump_path, force_path, spec = "none";
    int spec_max = kVerifyMaxTokens, eos = -1;
    int gen = 32;
    bool batched = false;
    EngineOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--pack") pack = next();
        else if (a == "--ids") ids_s = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--spec") spec = next();
        else if (a == "--spec-max") spec_max = std::stoi(next());
        else if (a == "--eos-id") eos = std::stoi(next());
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
        std::fprintf(stderr, "usage: ds41_generate --pack DIR --ids a,b,c [--gen N] [--threads T] [--dump F] [--force-ids F] [--spec none|suffix] [--spec-max 1..8] [--eos-id ID]\n");
        return 2;
    }
    std::vector<int> prompt = parse_ids(ids_s), forced;
    if (!force_path.empty()) {
        std::ifstream f(force_path);
        std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        forced = parse_ids(all);
    }
    try {
        // the teacher-forced path reads logits at the next forced id before the engine sees it
        for (const auto* ids : {&prompt, &forced})
            for (int t : *ids)
                if (t < 0 || t >= kVocab) throw std::invalid_argument("token id " + std::to_string(t) + " is outside the vocabulary");
        if (gen < 0 || spec_max < 1 || spec_max > 8 || (spec != "none" && spec != "suffix"))
            throw std::invalid_argument("use --gen >= 0, --spec none|suffix, --spec-max 1..8");
        if (spec == "suffix" && (!force_path.empty() || !dump_path.empty()))
            throw std::invalid_argument("--spec suffix cannot be combined with --force-ids or --dump");
        if (spec_max > kVerifyMaxTokens) {
            std::fprintf(stderr, "spec: cap T at %d until CPU rows 5..8 are validated\n", kVerifyMaxTokens);
            spec_max = kVerifyMaxTokens;
        }
        Engine engine(pack, opt);
        if (force_path.empty() && dump_path.empty()) {
            generate(engine, opt, prompt, gen, batched, spec == "suffix", spec_max, eos);
            return 0;
        }
        if (batched) {   // prefill the prompt (or the forced sequence) in one call, then decode with step()
            const std::vector<int>& pre = forced.empty() ? prompt : forced;
            std::vector<float> nll;
            int next = engine.prefill(pre, 0, forced.empty() ? nullptr : &nll);
            print_prefill(engine, pre.size());
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
        if (!dump_path.empty() && !dump) throw std::runtime_error("cannot open --dump " + dump_path);
        auto put = [&](const void* p, size_t size, size_t n) {
            if (std::fwrite(p, size, n, dump) != n) throw std::runtime_error("cannot write --dump " + dump_path);
        };
        StepDump sd;
        const int total = forced.empty() ? (int) prompt.size() + std::max(0, gen-1) : (int) forced.size();
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
            if (pos >= (int) prompt.size() - 1 && forced.empty() && gen > 0) out.push_back(next);
            std::fprintf(stderr,
                         "pos %d tok %d -> %d  total %.1f ms (engram reads %.1f, layers %.1f, of which cpu experts %.1f)"
                         "  vram hits %d/%d swaps %d  zero-copy %d cpu %d: ram %d file %d (ssd %d)  warmed %d useful %d\n",
                         pos, tok, next, t.total_ms, t.engram_ms, t.gpu_ms, t.cpu_experts_ms, t.expert_hits,
                         t.expert_total, t.vram_swaps, t.zero_copy_experts(), t.cpu_experts(), t.ram_experts, t.file_experts,
                         t.ssd_experts, t.warmed, t.warmed_useful);
            if (dump) {
                const int32_t hdr[2] = {tok, next};
                put(hdr, 4, 2);
                put(sd.hidden.data(), 2, sd.hidden.size());
                for (int l = 0; l < kLayers; ++l) put(sd.routes[l].data(), 4, kTopK);
                for (int l = 0; l < kLayers; ++l) put(sd.weights[l].data(), 4, kTopK);
                for (auto& p : sd.top_logits) put(&p.first, 4, 1);
                for (auto& p : sd.top_logits) put(&p.second, 4, 1);
            }
        }
        if (dump && std::fclose(dump) != 0) throw std::runtime_error("cannot write --dump " + dump_path);
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
