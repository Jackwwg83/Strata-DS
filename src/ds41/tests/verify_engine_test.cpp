// Each executable runs two isolated processes. Two packs need not fit in VRAM together.
#include "strata/ds41/config.hpp"
#include "strata/ds41/engine.hpp"
#include "strata/ds41/suffix_drafter.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::ds41;
#ifndef VERIFY_TEST
#define VERIFY_TEST "verify_rows_parity"
#endif
struct Case { int start, t, keep; };
static std::string pack, ids_path;
static EngineOptions options;
static bool exact = false, prefill_prefix = false;
static int token_at(int i, const std::vector<int>& ids) {
    return ids.empty() ? 10 + (i % 23) : ids.at(i % ids.size());
}
static void record(std::ofstream& f, int next, const std::vector<float>& logits) {
    f.write(reinterpret_cast<const char*>(&next), sizeof next);
    f.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
}
static void run(const Case c, bool verify, const std::string& path) {
    Engine e(pack, options);
    std::vector<int> ids;
    if (!ids_path.empty()) {
        std::ifstream f(ids_path);
        if (!f) throw std::runtime_error("cannot open IDs");
        std::string s((std::istreambuf_iterator<char>(f)), {});
        std::replace(s.begin(), s.end(), ',', ' ');
        std::istringstream in(s);
        int x; while (in >> x) ids.push_back(x);
        if (ids.empty()) throw std::runtime_error("empty IDs");
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open trace");
    int next = 0;
    SuffixDrafter suffix;
    std::vector<int> prefix(c.start);
    for (int p = 0; p < c.start; ++p) { prefix[p] = token_at(p, ids); suffix.append(prefix[p]); }
    if (prefill_prefix && !prefix.empty()) next = e.prefill(prefix, 0);
    else for (int p = 0; p < c.start; ++p) next = e.step(prefix[p], p);
    if (std::string(VERIFY_TEST) == "spec_end_to_end") {
        int p = c.start;
        suffix.append(next);
        std::vector<int> generated;
        for (int n = 0; n < 24;) {
            std::vector<int> window{next};
            int32_t draft[8];
            const int k = suffix.propose(std::min(c.t - 1, 24 - n - 1), draft);
            if (verify) window.insert(window.end(), draft, draft + k);
            if (verify) {
                auto result = e.verify(window, p);
                int keep = accepted_inputs(window, result.next);
                e.commit(keep);
                for (int j = 0; j < keep; ++j) {
                    generated.push_back(result.next[j]);
                    suffix.append(result.next[j]);
                }
                n += keep; p += keep; next = result.next[keep - 1];
            } else {
                next = e.step(next, p++); ++n;
                suffix.append(next); generated.push_back(next);
            }
        }
        f.write(reinterpret_cast<const char*>(generated.data()), generated.size() * sizeof(int));
        return;
    }
    // Repeat transactions to exercise graph replay and changing parity after partial acceptance.
    int pos = c.start, replays = 0;
    for (int round = 0; round < 8; ++round) {
        std::vector<int> window(c.t);
        for (int i = 0; i < c.t; ++i) window[i] = token_at(pos + i + round * 7, ids);
        if (verify) {
            auto must_throw = [](auto call) {
                bool rejected = false;
                try { call(); } catch (const std::exception&) { rejected = true; }
                if (!rejected) throw std::runtime_error("invalid transaction call was accepted");
            };
            must_throw([&] { e.commit(1); });
            must_throw([&] { e.verify({}, pos); });
            must_throw([&] { e.verify(std::vector<int>(kVerifyMaxTokens+1, 1), pos); });
            must_throw([&] { e.verify({-1}, pos); });
            must_throw([&] { e.verify({1}, pos+1); });
            auto result = e.verify(window, pos, true);
            replays += result.graph_reused;
            for (int i = 0; i < c.keep; ++i) record(f, result.next[i], result.logits[i]);
            must_throw([&] { e.step(window[0], pos); });
            must_throw([&] { e.prefill(window, pos); });
            must_throw([&] { e.verify(window, pos); });
            must_throw([&] { e.commit(0); });
            must_throw([&] { e.commit(c.t+1); });
            e.commit(c.keep);
            must_throw([&] { e.commit(c.keep); });
            if (e.last_logits() != result.logits[c.keep - 1]) throw std::runtime_error("commit logits row");
        } else {
            for (int i = 0; i < c.keep; ++i) {
                const int out = e.step(window[i], pos + i);
                record(f, out, e.last_logits());
            }
        }
        pos += c.keep;
        // Continue beyond another compressor group and Engram's n-gram tail.
        for (int i = 0; i < 5; ++i) {
            const int out = e.step(token_at(pos + 91, ids), pos);
            record(f, out, e.last_logits()); ++pos;
        }
    }
    if (!f) throw std::runtime_error("trace write failed");
    const char* graph = std::getenv("DS41_GRAPH");
    if (verify && (!graph || graph[0] != '0') && replays == 0)
        throw std::runtime_error("test did not reuse a verify graph");
    if (verify) { std::printf("verify_graph_replays %d\n", replays); std::fflush(stdout); }
}
static void child(const Case c, bool verify, const std::string& path) {
    const pid_t pid = fork();
    if (pid < 0) throw std::runtime_error("fork failed");
    if (pid == 0) {
        try { run(c, verify, path); _exit(0); }
        catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); _exit(1); }
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status))
        throw std::runtime_error("engine child failed");
}
static void compare(const std::string& a, const std::string& b, bool tokens_only, bool require_exact) {
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    size_t rows = 0, differing = 0;
    double max_abs = 0, sq = 0, refsq = 0;
    while (true) {
        int x = 0, y = 0;
        const bool ax = bool(fa.read(reinterpret_cast<char*>(&x), sizeof x));
        const bool by = bool(fb.read(reinterpret_cast<char*>(&y), sizeof y));
        if (ax != by) throw std::runtime_error("trace lengths differ");
        if (!ax) break;
        if (x != y) throw std::runtime_error("greedy tokens differ");
        ++rows;
        if (tokens_only) continue;
        std::vector<float> ra(kVocab), rb(kVocab);
        if (!fa.read(reinterpret_cast<char*>(ra.data()), kVocab * 4) ||
            !fb.read(reinterpret_cast<char*>(rb.data()), kVocab * 4)) throw std::runtime_error("short logits row");
        for (int i = 0; i < kVocab; ++i) {
            if (!std::isfinite(ra[i]) || !std::isfinite(rb[i])) throw std::runtime_error("nonfinite logit");
            differing += std::memcmp(&ra[i], &rb[i], 4) != 0;
            const double d = double(ra[i]) - rb[i];
            max_abs = std::max(max_abs, std::abs(d)); sq += d*d; refsq += double(ra[i])*ra[i];
        }
    }
    if (!rows) throw std::runtime_error("empty trace");
    const double rel = std::sqrt(sq / std::max(refsq, 1e-30));
    std::printf("rows=%zu differing=%zu max_abs=%.9g relative_l2=%.9g\n", rows, differing, max_abs, rel);
    // Same greedy IDs are mandatory. Grouped CPU reductions may change the last bits.
    if ((require_exact && differing) || max_abs > 0.05 || rel > 0.002)
        throw std::runtime_error("logits parity exceeded tolerance");
}
int main(int argc, char** argv) {
    try {
        int start = -1, t = 4;
        options.adapt_every = 0; options.vram_expert_slots = 0; options.ram_budget_gib = 0;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto value = [&]() { if (++i >= argc) throw std::runtime_error("missing option value"); return std::string(argv[i]); };
            if (a == "--pack") pack = value();
            else if (a == "--ids") ids_path = value();
            else if (a == "--start") start = std::stoi(value());
            else if (a == "--t") t = std::stoi(value());
            else if (a == "--threads") options.cpu_threads = std::stoi(value());
            else if (a == "--expert-profile") options.expert_profile = value();
            else if (a == "--vram-slots") options.vram_expert_slots = std::stoll(value());
            else if (a == "--ram-budget-gib") options.ram_budget_gib = std::stod(value());
            else if (a == "--max-seq") options.max_seq = std::stoi(value());
            else if (a == "--exact") exact = true;
            else if (a == "--prefill-prefix") prefill_prefix = true;
            else throw std::runtime_error("unknown argument: " + a);
        }
        if (pack.empty()) { std::puts("SKIP: pass --pack with a SAGE pack on a CUDA host"); return 77; }
        if (t < 1 || t > kVerifyMaxTokens) throw std::runtime_error("invalid T");
        setenv("DS41_ZC_QUOTA", "0", 1);
        const std::string name = VERIFY_TEST;
        std::vector<int> starts = {0, 1};
        if (name == "verify_ring_rollback") starts = {125, 127, 128};
        if (name == "verify_compressor") starts = {2, 3};
        if (name == "verify_engram") starts = {6, 127};
        if (name == "spec_end_to_end") starts = {8};
        if (start >= 0) starts = {start};
        char tmp[] = "/tmp/ds41-verify-XXXXXX";
        if (!mkdtemp(tmp)) throw std::runtime_error("mkdtemp failed");
        const std::string a = std::string(tmp) + "/baseline", b = std::string(tmp) + "/verify";
        for (int p : starts) for (int m = 1; m <= t; ++m) {
            const int first = name == "verify_rows_parity" || name == "spec_end_to_end" ? m : 1;
            for (int keep = first; keep <= m; ++keep) {
                std::printf("case %s start=%d T=%d keep=%d\n", VERIFY_TEST, p, m, keep); std::fflush(stdout);
                options.max_seq = std::max(options.max_seq, p + 128);
                child({p, m, keep}, false, a); child({p, m, keep}, true, b);
                compare(a, b, name == "spec_end_to_end", exact || m == 1);
            }
        }
        std::remove(a.c_str()); std::remove(b.c_str()); rmdir(tmp);
        std::printf("RESULT pass %s\n", VERIFY_TEST);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "RESULT fail %s: %s\n", VERIFY_TEST, e.what()); return 1; }
}
