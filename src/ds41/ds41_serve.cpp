// src/ds41/ds41_serve.cpp - DeepSeek V4.1 Flash behind serve/server.py: upstream's `strata --serve` line protocol
// (src/program/generate.cpp) on the ds41 engine.
//
//   ds41_serve --serve --pack DIR --max-context N [--expert-profile F] [--ram-budget-gib G] [--threads T] ...
//
// stdin:  GEN <max_new> [key=value ...] id,id,...   one request; the sampling keys are upstream's
//         STOP                                       ends the running request (read on its own thread)
//         QUIT                                       ends the engine
// stdout: INFO k=v ...  then  READY <context> stop   once, when the model is loaded
//         RESUME <reused>                            per request: the session tokens the prompt reuses
//         PP <done> <total> <ms> <tok/s>             prompt progress (every layer of a pass): also the heartbeat
//         T <id>                                     each output token
//         DONE ... | ERR <message>                   the end of a request (DONE: serve_request.hpp)
// Not supported, answered with ERR: GENI (images), batch slots (BGEN, BSTOP, BYIELD), VRAM.
//
// The session is the tokens fed so far. A prompt that starts with all of them reads only the rest; any other
// prompt starts over at position 0 (the engine cannot go back to an earlier position yet).
#include "strata/ds41/config.hpp"
#include "strata/ds41/engine.hpp"
#include "strata/ds41/serve_request.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace strata::ds41;
using strata::ds41::serve::DoneStats;
using strata::ds41::serve::Request;

namespace {

constexpr int kPenaltyWindowCap = 4096;   // upstream's cap on penalty_last_n

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// One protocol line on stdout, flushed (stdout is a pipe: fully buffered otherwise)
void say(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41_serve: ") + what + ": " + cudaGetErrorString(e));
}

/// Upstream's sampler chain (penalties -> top_k -> top_p -> min_p -> temperature -> pick) on the last logits
class Sampler {
public:
    Sampler() {
        ck(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking), "sampler stream");
        ck(cudaMalloc(&logits_, (size_t) kVocab * 4), "sampler logits");
        ck(cudaMalloc(&hist_, (size_t) kPenaltyWindowCap * 4), "sampler history");
        ck(cudaMalloc(&out_, 4), "sampler out");
    }
    ~Sampler() {
        cudaFree(logits_);
        cudaFree(hist_);
        cudaFree(out_);
        cudaStreamDestroy(st_);
    }
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    /// The token after `session` (every token fed, most recent last), from its logits
    int pick(const std::vector<float>& lg, const Request& r, const std::vector<int>& session, uint64_t seed,
             uint64_t counter) {
        strata::kernels::SamplerParams p;
        p.greedy = r.temperature <= 0.0f;
        p.temperature = r.temperature;
        p.top_p = r.top_p;
        p.top_k = r.top_k;
        p.min_p = std::clamp(r.min_p, 0.0f, 1.0f);
        p.seed = seed;
        p.counter = counter;
        const int h = std::min({std::max(r.penalty_last_n, 0), kPenaltyWindowCap, (int) session.size()});
        p.penalty_last_n = h;
        p.penalty_repeat = r.penalty_repeat;
        p.penalty_freq = r.penalty_freq;
        p.penalty_present = r.penalty_present;
        ck(cudaMemcpyAsync(logits_, lg.data(), (size_t) kVocab * 4, cudaMemcpyHostToDevice, st_), "logits up");
        if (h > 0)
            ck(cudaMemcpyAsync(hist_, session.data() + session.size() - h, (size_t) h * 4, cudaMemcpyHostToDevice, st_),
               "history up");
        strata::kernels::sample_tokens(logits_, 1, kVocab, h > 0 ? hist_ : nullptr, h, p, out_, st_);
        int tok = -1;
        ck(cudaMemcpyAsync(&tok, out_, 4, cudaMemcpyDeviceToHost, st_), "pick down");
        ck(cudaStreamSynchronize(st_), "sample");
        if (tok < 0 || tok >= kVocab) throw std::runtime_error("ds41_serve: the sampler returned no token");
        return tok;
    }

private:
    cudaStream_t st_ = nullptr;
    float* logits_ = nullptr;
    int* hist_ = nullptr;
    int* out_ = nullptr;
};

/// stdin on its own thread (upstream): STOP is seen while a request runs, everything else is queued
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};

    void start() {
        std::thread([this] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") {
                    stop.store(true);
                    continue;
                }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        }).detach();
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !lines.empty() || eof; });
        if (lines.empty()) return false;
        out = std::move(lines.front());
        lines.pop_front();
        return true;
    }
};

struct Options {
    std::string pack;
    int max_context = 8192;
    int step_prompt_max = 8;   ///< a prompt part this short is read token by token (a pass costs ~300 ms however short)
    std::vector<int> eos = {1};   ///< <｜end▁of▁sentence｜>
    EngineOptions eng;
};

[[noreturn]] void usage(const std::string& why) {
    std::fprintf(stderr,
                 "ds41_serve: %s\n"
                 "usage: ds41_serve --serve --pack DIR [--max-context N] [--threads T] [--expert-profile F]\n"
                 "       [--vram-slots N] [--ram-budget-gib G] [--adapt-every N] [--adapt-swaps N] [--prefill-chunk N]\n"
                 "       [--prefill-batch N] [--prefill-ring N] [--prefill-threads N] [--step-prompt-max N]\n"
                 "       [--eos-id ID ...]\n",
                 why.c_str());
    std::exit(2);
}

Options parse_args(int argc, char** argv) {
    Options o;
    bool serve = false, eos_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage("missing value for " + a);
            return argv[++i];
        };
        auto num = [&](const std::string& v) -> double {
            try {
                size_t used = 0;
                const double d = std::stod(v, &used);
                if (used != v.size()) throw std::invalid_argument(v);
                return d;
            } catch (const std::exception&) {
                usage(a + " needs a number, not '" + v + "'");
            }
        };
        if (a == "--serve") serve = true;
        else if (a == "--pack") o.pack = next();
        else if (a == "--max-context") o.max_context = (int) num(next());
        else if (a == "--threads") o.eng.cpu_threads = (int) num(next());
        else if (a == "--expert-profile") o.eng.expert_profile = next();
        else if (a == "--vram-slots") o.eng.vram_expert_slots = (int64_t) num(next());
        else if (a == "--ram-budget-gib") o.eng.ram_budget_gib = num(next());
        else if (a == "--adapt-every") o.eng.adapt_every = (int) num(next());
        else if (a == "--adapt-swaps") o.eng.adapt_swaps = (int) num(next());
        else if (a == "--prefill-chunk") o.eng.prefill_chunk = (int) num(next());
        else if (a == "--prefill-batch") o.eng.prefill_batch = (int) num(next());
        else if (a == "--prefill-ring") o.eng.prefill_ring = (int) num(next());
        else if (a == "--prefill-threads") o.eng.prefill_threads = (int) num(next());
        else if (a == "--step-prompt-max") o.step_prompt_max = (int) num(next());
        else if (a == "--eos-id") {
            if (!eos_given) o.eos.clear();
            eos_given = true;
            o.eos.push_back((int) num(next()));
        } else usage("unknown argument " + a + " (this engine has no " + a + ")");
    }
    if (!serve) usage("only the --serve mode exists");
    if (o.pack.empty()) usage("--pack is required");
    if (o.max_context < 64) usage("--max-context must be at least 64");
    o.eng.max_seq = o.max_context;
    return o;
}

/// The session: every token the engine was fed, in order (engine.position() == size())
struct Session {
    Engine& e;
    std::vector<int> live;

    void restart() {
        e.reset();
        live.clear();
    }
};

}  // namespace

int main(int argc, char** argv) {
    const Options o = parse_args(argc, argv);
    std::unique_ptr<Engine> engine;
    std::unique_ptr<Sampler> sampler;
    try {
        engine = std::make_unique<Engine>(o.pack, o.eng);
        sampler = std::make_unique<Sampler>();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "ds41_serve: the model did not load: %s\n", ex.what());
        return 1;
    }
    Engine& e = *engine;
    Session s{e, {}};
    Input in;
    in.start();
    say("INFO context=%d expert_slots=%d spec=0 model=deepseek-v4.1-flash engine=" STRATA_VERSION, o.max_context,
        e.vram_expert_slots());
    say("READY %d stop", o.max_context);   // "stop": this engine honours STOP

    std::string line;
    while (in.next(line)) {
        if (line == "QUIT") break;
        if (line.empty()) continue;
        in.stop.store(false);   // a STOP that arrived between requests is stale
        if (line == "VRAM" || line.rfind("VRAM ", 0) == 0) {
            say("ERR VRAM is not supported by ds41_serve");
            continue;
        }
        if (line.rfind("GENI ", 0) == 0) {
            say("ERR this engine has no image input");
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) {
            say("ERR expected: GEN <max_new> <id,id,...> (ds41_serve has no batch slots)");
            continue;
        }
        Request r;
        const std::string bad = serve::parse_gen(line, r);
        if (!bad.empty()) {
            say("ERR bad request: %s", bad.c_str());
            continue;
        }
        const long long n = (long long) r.ids.size();
        if (n + r.max_new + 8 > o.max_context) {
            say("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%d)", n, r.max_new, o.max_context);
            continue;
        }
        if (std::any_of(r.ids.begin(), r.ids.end(), [](int t) { return t < 0 || t >= kVocab; })) {
            say("ERR a token id is outside the vocabulary");
            continue;
        }

        DoneStats d;
        d.prompt = n;
        const double r0 = now_ms();
        const uint64_t seed = r.seed ? r.seed : (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
        try {
            // ---- the prompt: reuse the session when the prompt continues it, else start over
            const int64_t resume = serve::reusable(s.live, r.ids);
            if (resume == 0 && !s.live.empty()) s.restart();
            d.reused = resume;
            say("RESUME %lld", (long long) resume);
            const std::vector<int> rest(r.ids.begin() + resume, r.ids.end());
            const double pp0 = now_ms();
            auto pp = [&](int done) {
                const double ms = now_ms() - pp0;
                say("PP %lld %lld %.0f %.1f", (long long) (resume + done), n, ms,
                    ms > 0 ? 1000.0 * done / ms : 0.0);
            };
            int greedy_next = -1;
            bool cancelled = false;
            int read = 0;
            if ((int) rest.size() <= o.step_prompt_max) {
                for (int t : rest) {
                    greedy_next = e.step(t, e.position());
                    s.live.push_back(t);
                    pp(++read);
                    if (in.stop.load() && read < (int) rest.size()) {
                        cancelled = true;
                        break;
                    }
                }
            } else {
                e.set_prefill_progress([&](int done, int) {
                    pp(done);
                    return !in.stop.load();
                });
                try {
                    greedy_next = e.prefill(rest, e.position());
                    read = (int) rest.size();
                    s.live.insert(s.live.end(), rest.begin(), rest.end());
                } catch (const PrefillCancelled&) {
                    s.live.clear();   // the engine reset itself
                    cancelled = true;
                }
                e.set_prefill_progress(nullptr);
            }
            d.read = read;
            d.prompt_ms = now_ms() - r0;

            // ---- the output: the last prompt token's logits pick the first token; each later one is fed, then picked
            const double t0 = now_ms();
            if (cancelled) {
                d.finish = "cancel";
            } else {
                uint64_t counter = 0;
                int tok = r.greedy() ? greedy_next : sampler->pick(e.last_logits(), r, s.live, seed, counter++);
                for (;;) {
                    say("T %d", tok);
                    ++d.generated;
                    if (std::find(o.eos.begin(), o.eos.end(), tok) != o.eos.end()) {
                        d.finish = "stop";
                        break;
                    }
                    if (d.generated >= r.max_new) {
                        d.finish = "length";
                        break;
                    }
                    if (in.stop.load()) {
                        d.finish = "cancel";
                        break;
                    }
                    greedy_next = e.step(tok, e.position());
                    s.live.push_back(tok);
                    const Engine::Timing& tm = e.last_timing();
                    d.hits += tm.expert_hits;
                    d.lookups += tm.expert_total;
                    d.ram += tm.ram_experts;
                    d.file += tm.file_experts;
                    d.offloaded += tm.zero_copy_experts();
                    tok = r.greedy() ? greedy_next : sampler->pick(e.last_logits(), r, s.live, seed, counter++);
                }
            }
            d.decode_ms = now_ms() - t0;
            say("%s", serve::done_line(d).c_str());
            std::fprintf(stderr,
                         "ds41_serve: prompt %lld (reused %lld, read %lld) %.0f ms, %lld out %.1f ms/token, "
                         "VRAM hits %lld of %lld, %s\n",
                         n, d.reused, d.read, d.prompt_ms, d.generated,
                         d.generated > 1 ? d.decode_ms / (double) (d.generated - 1) : 0.0, d.hits, d.lookups,
                         d.finish.c_str());
            std::fflush(stderr);
        } catch (const std::exception& ex) {
            e.set_prefill_progress(nullptr);
            say("ERR %s", ex.what());
            try {
                s.restart();   // the next request starts over at position 0
            } catch (const std::exception& again) {
                // the engine refuses every call after a failure that left its state half written: end, so the
                // server starts a new one
                std::fprintf(stderr, "ds41_serve: the engine is unusable (%s); exiting\n", again.what());
                return 1;
            }
        }
    }
    return 0;
}
