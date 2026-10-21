// src/ds41/ds41_serve.cpp - DeepSeek V4.1 Flash behind serve/server.py: upstream's `strata --serve` line protocol
// (src/program/generate.cpp) on the ds41 engine.
//
//   ds41_serve --serve --pack DIR --max-context N [--expert-profile F] [--ram-budget-gib G] [--threads T] ...
//
// stdin:  GEN <max_new> [key=value ...] id,id,...   one request; the sampling keys are upstream's
//         STOP                                       ends the running request (read on its own thread)
//         BGEN <slot> <max_new> [keys] ids           with --batch N: a request for a batch slot (upstream's)
//         BSTOP <slot>                               ends a slot's request
//         QUIT                                       ends the engine
// stdout: INFO k=v ...  then  READY <context> stop   once, when the model is loaded (batch_slots=N with --batch)
//         RESUME <reused>                            per request: the session tokens the prompt reuses
//         PP <done> <total> <ms> <tok/s>             prompt progress (every layer of a pass): also the heartbeat
//         T <id>                                     each output token
//         DONE ... | ERR <message>                   the end of a request (DONE: serve_request.hpp)
//         BADM <slot> 1|0                            after a BGEN's DONE (or ERR): it continues in the slot, or not
//         BT <slot> <id>, BDONE <slot> <n> <finish> <ms>   the slots' tokens, decoded together between lines
// Not supported, answered with ERR: GENI and BGENI (images), VRAM. BYIELD is ignored (upstream allows that).
//
// The session is the tokens fed so far. A prompt that starts with all of them reads only the rest. Before each
// prompt's last token the engine keeps a snapshot (--snapshots slots, least recently used first); a prompt that starts
// with a snapshot's tokens goes back there, and one that continues an idle batch slot's conversation takes that slot's
// state (upstream's slot_cache). Any other prompt starts over at position 0. A BGEN is read in the main session (the
// slots wait), then its state is copied into the slot.
#include "strata/ds41/config.hpp"
#include "strata/ds41/draft_scheduler.hpp"
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
        return pick(lg.data(), r, session, seed, counter);
    }
    int pick(const float* lg, const Request& r, const std::vector<int>& session, uint64_t seed, uint64_t counter) {
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
        ck(cudaMemcpyAsync(logits_, lg, (size_t) kVocab * 4, cudaMemcpyHostToDevice, st_), "logits up");
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
    /// what the reader thread touches: shared with it, never main's locals, since it is detached and may sit in
    /// read(2) past QUIT (upstream 0.1.44)
    struct Shared {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> lines;
        bool eof = false;
        std::atomic<bool> stop{false};
    };
    std::shared_ptr<Shared> sh = std::make_shared<Shared>();
    std::atomic<bool>& stop = sh->stop;

    void start() {
        std::thread([st = sh] {
            // read(2) on the descriptor, not std::cin: glibc's exit() flushes every stdio stream and waits for stdin's
            // lock, which getline holds while it waits for input, so the process hung at its end until stdin closed
            std::string l, buf;
            char chunk[4096];
            auto getline_fd = [&](std::string& out) -> bool {
                for (;;) {
                    const size_t nl = buf.find('\n');
                    if (nl != std::string::npos) {
                        out.assign(buf, 0, nl);
                        buf.erase(0, nl + 1);
                        return true;
                    }
                    const ssize_t n = ::read(0, chunk, sizeof chunk);
                    if (n < 0 && errno == EINTR) continue;
                    if (n <= 0) {
                        if (buf.empty()) return false;
                        out.swap(buf);
                        buf.clear();
                        return true;
                    }
                    buf.append(chunk, (size_t) n);
                }
            };
            while (getline_fd(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") {
                    st->stop.store(true);
                    continue;
                }
                std::lock_guard<std::mutex> lk(st->mu);
                st->lines.push_back(l);
                st->cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(st->mu);
            st->eof = true;
            st->cv.notify_one();
        }).detach();
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(sh->mu);
        sh->cv.wait(lk, [&] { return !sh->lines.empty() || sh->eof; });
        if (sh->lines.empty()) return false;
        out = std::move(sh->lines.front());
        sh->lines.pop_front();
        return true;
    }
    /// a waiting line, without blocking (batch slots decode while nothing arrives)
    bool poll(std::string& out) {
        std::lock_guard<std::mutex> lk(sh->mu);
        if (sh->lines.empty()) return false;
        out = std::move(sh->lines.front());
        sh->lines.pop_front();
        return true;
    }
    bool ended() {
        std::lock_guard<std::mutex> lk(sh->mu);
        return sh->eof && sh->lines.empty();
    }
};

struct Options {
    std::string pack;
    int max_context = 8192;
    /// a prompt part this short is read in verify windows of 4 tokens (upstream: a short part goes through the
    /// verify windows); a pass streams every expert outside VRAM over PCIe, which costs seconds however short it is.
    /// RTX 4090, SAGE 1.59bpw (ds41/bench/scripts/prompt_window_ab.py): windows ~32 ms per token, a pass 2.6 s plus
    /// ~4.6 ms per token; equal at about 125 tokens (128 new: 3.9 s both).
    int window_prompt_max = 120;
    std::vector<int> eos = {1};   ///< <｜end▁of▁sentence｜>
    /// --spec dspark: requests decode in DSpark's verify windows. Greedy: the same tokens as plain steps. Sampling: each
    /// row sampled with the plain path's chain, counter and history, a draft kept only when it equals the pick (upstream's
    /// exact-match verify), so the text depends on the seed, not on the drafts.
    /// --draft-k K: K drafts a round (0..3); without it DraftScheduler chooses, and drafts only while drafting pays.
    bool dspark = false;
    int draft_k = -1;
    bool draft_always = false;   ///< --draft-always: DraftScheduler::always (every round drafts, k by confidence)
    /// --prompt-cache-root N (0: off): a prompt read through the end of a system prompt of N tokens or more (its first
    /// --turn-token, DeepSeek's <｜User｜>) keeps the engine's root checkpoint there (Engine::save_root), and a later
    /// prompt that starts with it - a new conversation of the same agent client - starts reading after it (upstream's
    /// system-prompt checkpoint, --prompt-cache-root, engine 0.1.20)
    int root_min = 2048;
    int turn_token = 128803;
    /// --root-file PATH: the root checkpoints on disk, read at the start and written when the engine is idle after a
    /// new one, so a restart keeps them (a new conversation of the agent client starts after its system prompt at once)
    std::string root_file;
    /// --roots N: root checkpoint slots (EngineOptions::roots, least recently used first). Each client's system prompt
    /// keeps one, and with 2 or more a new conversation also keeps one where it stops sharing tokens with one of the
    /// last new conversations (--root-deep N tokens or more past what it reuses: an agent client's first message
    /// repeats its context files). Each root takes about 37 KB a token.
    int root_deep = 512;
    EngineOptions eng;
};

[[noreturn]] void usage(const std::string& why) {
    std::fprintf(stderr,
                 "ds41_serve: %s\n"
                 "usage: ds41_serve --serve --pack DIR [--max-context N] [--threads T] [--expert-profile F]\n"
                 "       [--vram-slots N] [--vram-reserve-mib M] [--ram-budget-gib G] [--adapt-every N] [--adapt-swaps N] [--prefill-chunk N]\n"
                 "       [--prefill-batch N] [--prefill-ring N] [--prefill-threads N] [--window-prompt-max N]\n"
                 "       [--snapshots N] [--batch N] [--spec none|dspark] [--draft-k K] [--draft-always]\n"
                 "       [--prompt-cache-root N] [--turn-token ID] [--root-file PATH] [--roots N] [--root-deep N]\n"
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
        else if (a == "--vram-reserve-mib") o.eng.vram_reserve_bytes = (size_t) num(next()) << 20;
        else if (a == "--ram-budget-gib") o.eng.ram_budget_gib = num(next());
        else if (a == "--adapt-every") o.eng.adapt_every = (int) num(next());
        else if (a == "--adapt-swaps") o.eng.adapt_swaps = (int) num(next());
        else if (a == "--prefill-chunk") o.eng.prefill_chunk = (int) num(next());
        else if (a == "--prefill-batch") o.eng.prefill_batch = (int) num(next());
        else if (a == "--prefill-ring") o.eng.prefill_ring = (int) num(next());
        else if (a == "--prefill-threads") o.eng.prefill_threads = (int) num(next());
        else if (a == "--window-prompt-max") o.window_prompt_max = (int) num(next());
        else if (a == "--snapshots") o.eng.snapshots = (int) num(next());
        else if (a == "--batch") o.eng.batch_slots = (int) num(next());
        else if (a == "--spec") {
            const std::string v = next();
            if (v != "none" && v != "dspark") usage("--spec is none or dspark");
            o.dspark = v == "dspark";
        } else if (a == "--draft-k") o.draft_k = (int) num(next());
        else if (a == "--draft-always") o.draft_always = true;
        else if (a == "--prompt-cache-root") o.root_min = (int) num(next());
        else if (a == "--turn-token") o.turn_token = (int) num(next());
        else if (a == "--root-file") o.root_file = next();
        else if (a == "--roots") o.eng.roots = (int) num(next());
        else if (a == "--root-deep") o.root_deep = (int) num(next());
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
    if (o.draft_k > kVerifyMaxTokens - 1 || (o.draft_k >= 0 && !o.dspark)) usage("--draft-k 0..3 needs --spec dspark");
    o.eng.dspark = o.dspark;
    return o;
}

/// The session: every token the engine was fed, in order (engine.position() == size())
struct Session {
    Engine& e;
    std::vector<int> live;
    /// per engine snapshot slot: the tokens fed before its position (empty: unused), and when it was last used
    std::vector<std::vector<int>> snaps;
    std::vector<uint64_t> used;
    uint64_t clock = 0;

    void restart() {
        e.reset();
        live.clear();
    }
    /// a snapshot of the current position in the least recently used slot (none without slots)
    void save() {
        if (snaps.empty()) return;
        const size_t slot = (size_t) (std::min_element(used.begin(), used.end()) - used.begin());
        snaps[slot].clear();   // a failed save leaves the slot empty
        e.save_snapshot((int) slot);
        snaps[slot] = live;
        used[slot] = ++clock;
    }
    /// back to slot's position: the session ends there
    int64_t restore(int slot) {
        const int pos = e.restore_snapshot(slot);
        live.resize((size_t) pos);
        used[(size_t) slot] = ++clock;
        return pos;
    }
};

/// a batch slot's request (upstream's BSlot)
struct Slot {
    bool active = false, stop = false;
    bool cached = false;         ///< idle, and its engine state holds `ids` (a later turn can continue there)
    int x = -1;                  ///< the token to feed next (already written as T or BT)
    long long produced = 0, max_new = 0;
    Request r;                   ///< the sampling keys
    uint64_t seed = 0, counter = 0;
    std::vector<int> ids;        ///< the tokens fed to the slot
    double t0 = 0;               ///< when the request was admitted
};

/// how a request in the main session ended
struct Outcome {
    enum { kErr, kDone, kFatal } status = kErr;   ///< kErr: ERR written; kFatal: the engine is unusable
    std::string finish;
    long long generated = 0;
    int last = -1;               ///< the last token written
    uint64_t seed = 0, counter = 0;
    double t0 = 0;
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
    Session s{e, {}, std::vector<std::vector<int>>((size_t) e.snapshot_slots()),
              std::vector<uint64_t>((size_t) e.snapshot_slots(), 0)};
    Input in;
    in.start();
    const int nslots = e.batch_slots();
    std::vector<Slot> slots((size_t) nslots);
    const std::string batch_info =
        nslots >= 2 ? " batch_slots=" + std::to_string(nslots) + " slot_cache=1" : std::string();
    // the root checkpoints: when each slot was last used, the last new conversations' prompts, and whether a root
    // is newer than the file
    std::vector<uint64_t> root_used((size_t) e.root_slots(), 0);
    uint64_t root_clock = 0;
    std::vector<std::vector<int>> recent_new;
    bool roots_dirty = false;
    if (!o.root_file.empty() && o.root_min > 0) {
        const double t0 = now_ms();
        const int got = e.read_root(o.root_file);
        for (int i = 0; i < got; ++i) {
            root_used[(size_t) i] = ++root_clock;
            std::fprintf(stderr, "ds41_serve: root checkpoint %d at %zu tokens read from %s\n", i,
                         e.root_tokens(i).size(), o.root_file.c_str());
        }
        if (got) std::fprintf(stderr, "ds41_serve: %d root checkpoints read in %.1f s\n", got, (now_ms() - t0) / 1000.0);
    }
    // the session's tokens so far as a root, in the least recently used slot. A failed save (no memory for it) only
    // costs the reuse, so the request goes on.
    auto keep_root = [&](const std::vector<int>& tokens) {
        if (root_used.empty()) return;
        const int slot = (int) (std::min_element(root_used.begin(), root_used.end()) - root_used.begin());
        try {
            e.save_root(slot, tokens);
            root_used[(size_t) slot] = ++root_clock;
            roots_dirty = true;
            std::fprintf(stderr, "ds41_serve: root checkpoint %d at %zu tokens\n", slot, tokens.size());
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "ds41_serve: no root checkpoint at %zu tokens (%s)\n", tokens.size(), ex.what());
        }
    };
    auto write_roots = [&] {
        if (!roots_dirty || o.root_file.empty()) return;
        roots_dirty = false;
        const double t0 = now_ms();
        try {
            e.write_root(o.root_file);
            std::fprintf(stderr, "ds41_serve: root checkpoints written to %s in %.1f s\n", o.root_file.c_str(),
                         (now_ms() - t0) / 1000.0);
        } catch (const std::exception& ex) {   // the next start reads the older file, or none
            std::fprintf(stderr, "ds41_serve: the root checkpoints were not written (%s)\n", ex.what());
        }
    };
    DraftScheduler sched;   // DSpark: one policy for the process (its step and verify times carry over requests)
    sched.always = o.draft_always;
    say("INFO context=%d expert_slots=%d spec=0 model=deepseek-v4.1-flash engine=" STRATA_VERSION "%s", o.max_context,
        e.vram_expert_slots(), batch_info.c_str());
    say("READY %d stop", o.max_context);   // "stop": this engine honours STOP

    // One request in the main session: its prompt (reusing the session, a snapshot or an idle slot's conversation),
    // then up to r.max_new tokens (T lines) and the DONE line. ERR on a bad request or an engine failure.
    auto serve_gen = [&](const Request& r) -> Outcome {
        Outcome oc;
        const long long n = (long long) r.ids.size();
        if (n + r.max_new + 8 > o.max_context) {
            say("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%d)", n, r.max_new, o.max_context);
            return oc;
        }
        if (std::any_of(r.ids.begin(), r.ids.end(), [](int t) { return t < 0 || t >= kVocab; })) {
            say("ERR a token id is outside the vocabulary");
            return oc;
        }
        DoneStats d;
        d.prompt = n;
        long long d_rounds = 0, d_drafted = 0, d_accepted = 0;   // DSpark's verify windows of this request
        const double r0 = now_ms();
        oc.seed = r.seed ? r.seed : (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
        try {
            // ---- the prompt: reuse the session when the prompt continues it, or go back to a snapshot that the
            // prompt starts with (DeepSeek drops the last answer's reasoning, so the next prompt differs from the
            // session at that answer's start: the snapshot before the last prompt token is there), or take the
            // conversation an idle batch slot holds (upstream's slot_cache), whichever reuses most; else start over
            int64_t resume = serve::reusable(s.live, r.ids);
            const int snap = serve::pick_snapshot(s.snaps, s.live, r.ids, resume);
            int64_t best = std::max<int64_t>(resume, snap >= 0 ? (int64_t) s.snaps[(size_t) snap].size() : 0);
            int from_slot = -1;
            for (int b = 0; b < nslots; ++b) {
                const Slot& sl = slots[(size_t) b];
                if (sl.active || !sl.cached) continue;
                const int64_t len = serve::reusable(sl.ids, r.ids);
                if (len > best) {
                    best = len;
                    from_slot = b;
                }
            }
            // a root checkpoint: the longest one this prompt starts with (a system prompt's end, or deeper)
            std::vector<std::vector<int>> root_toks((size_t) e.root_slots());
            for (int i = 0; i < e.root_slots(); ++i) root_toks[(size_t) i] = e.root_tokens(i);
            const int from_root = o.root_min > 0 ? serve::pick_root(root_toks, r.ids, best) : -1;
            if (from_root >= 0) {
                resume = e.restore_root(from_root);
                s.live = root_toks[(size_t) from_root];
                root_used[(size_t) from_root] = ++root_clock;
            } else if (from_slot >= 0) {
                e.copy_from_slot(from_slot);
                s.live = slots[(size_t) from_slot].ids;
                resume = best;
            } else if (snap >= 0) {
                resume = s.restore(snap);
            } else if (resume == 0 && !s.live.empty()) {
                s.restart();
            }
            // a new conversation: from a root or from position 0, not a later turn of one
            const bool fresh = from_root >= 0 || (from_slot < 0 && snap < 0 && resume == 0);
            d.reused = resume;
            say("RESUME %lld", (long long) resume);
            const std::vector<int> rest(r.ids.begin() + resume, r.ids.end());
            const double pp0 = now_ms();
            int read = 0;
            auto pp = [&](int done) {
                const double ms = now_ms() - pp0;
                say("PP %lld %lld %.0f %.1f", (long long) (resume + done), n, ms,
                    ms > 0 ? 1000.0 * done / ms : 0.0);
            };
            int greedy_next = -1;
            bool cancelled = false;
            // part of the prompt: in verify windows of 4 tokens, or one batched prefill; false when STOP ended it
            auto feed = [&](std::vector<int>::const_iterator b, std::vector<int>::const_iterator end,
                            bool windows) -> bool {
                const std::vector<int> part(b, end);
                if (windows) {
                    for (size_t i = 0; i < part.size(); i += kVerifyMaxTokens) {
                        const std::vector<int> win(part.begin() + i,
                                                   part.begin() + std::min(part.size(), i + kVerifyMaxTokens));
                        const VerifyResult v = e.verify(win, e.position());
                        e.commit((int) win.size());   // all of them: the window is prompt, not drafts
                        greedy_next = v.next.back();
                        s.live.insert(s.live.end(), win.begin(), win.end());
                        read += (int) win.size();
                        pp(read);
                        if (in.stop.load() && read < (int) rest.size()) return false;
                    }
                    return true;
                }
                const int before = read;
                e.set_prefill_progress([&](int done, int) {
                    pp(before + done);
                    return !in.stop.load();
                });
                try {
                    greedy_next = e.prefill(part, e.position());
                    read += (int) part.size();
                    s.live.insert(s.live.end(), part.begin(), part.end());
                } catch (const PrefillCancelled&) {
                    s.live.clear();   // the engine reset itself
                    e.set_prefill_progress(nullptr);
                    return false;
                }
                e.set_prefill_progress(nullptr);
                return true;
            };
            // read through the end of a long system prompt and keep a root there, then (a new conversation) through
            // what it shares with one of the last new conversations and keep a deeper root there
            size_t from = 0;
            auto cut_root = [&](int64_t at) {
                cancelled = !feed(rest.begin() + (std::ptrdiff_t) from, rest.begin() + (at - resume), false);
                if (cancelled) return;
                keep_root(std::vector<int>(r.ids.begin(), r.ids.begin() + at));
                from = (size_t) (at - resume);
            };
            if (o.root_min > 0) {
                const auto turn = std::find(r.ids.begin(), r.ids.end(), o.turn_token);
                const int64_t b = turn - r.ids.begin();
                if (turn != r.ids.end() && b >= o.root_min && b > resume && b + 1 < n) cut_root(b);
                if (!cancelled && fresh && o.root_deep > 0 && e.root_slots() >= 2) {
                    int64_t deep = 0;
                    for (const std::vector<int>& prev : recent_new)
                        deep = std::max(deep, serve::deep_checkpoint(prev, r.ids, resume + (int64_t) from, o.root_deep));
                    if (deep > 0) cut_root(deep);
                }
                if (!cancelled && fresh) {
                    recent_new.push_back(r.ids);
                    if (recent_new.size() > 4) recent_new.erase(recent_new.begin());
                }
            }
            const auto rb = rest.begin() + (std::ptrdiff_t) from;
            const bool windows = (int) (rest.end() - rb) <= o.window_prompt_max;
            if (cancelled) {
            } else if (rest.end() - rb >= 2 && !s.snaps.empty()) {
                // all but the last token, a snapshot, then the last token (one window)
                cancelled = !feed(rb, rest.end() - 1, windows);
                if (!cancelled) {
                    s.save();
                    cancelled = !feed(rest.end() - 1, rest.end(), true);
                }
            } else {
                cancelled = !feed(rb, rest.end(), windows);
            }
            d.read = read;
            d.prompt_ms = now_ms() - r0;

            // ---- the output: the last prompt token's logits pick the first token; each later one is fed, then picked
            const double t0 = now_ms();
            if (cancelled) {
                d.finish = "cancel";
            } else {
                auto add_timing = [&] {
                    const Engine::Timing& tm = e.last_timing();
                    d.hits += tm.expert_hits;
                    d.lookups += tm.expert_total;
                    d.ram += tm.ram_experts;
                    d.file += tm.file_experts;
                    d.offloaded += tm.zero_copy_experts();
                };
                // one output token: the T line, then whether the request ends there (stop, length)
                auto emit = [&](int t) -> bool {
                    say("T %d", t);
                    oc.last = t;
                    ++d.generated;
                    if (std::find(o.eos.begin(), o.eos.end(), t) != o.eos.end()) {
                        d.finish = "stop";
                        return true;
                    }
                    if (d.generated >= r.max_new) {
                        d.finish = "length";
                        return true;
                    }
                    return false;
                };
                int tok = r.greedy() ? greedy_next : sampler->pick(e.last_logits(), r, s.live, oc.seed, oc.counter++);
                bool ended = emit(tok);
                while (!ended) {
                    if (in.stop.load()) {
                        d.finish = "cancel";
                        break;
                    }
                    // tok is written and not fed. DSpark: a verify window of tok and up to 3 drafts, never past
                    // max_new (the round writes at most k + 1 tokens) or the context
                    const long long room = std::min<long long>(r.max_new - d.generated,
                                                               (long long) o.max_context - e.position() - 2);
                    if (o.dspark && room >= 2 && (o.draft_k >= 0 || sched.want_draft())) {
                        const double r0 = now_ms();
                        const Engine::Draft dd = e.draft(tok);
                        const int cap = (int) std::min<long long>(room - 1, kVerifyMaxTokens - 1);
                        const int k = o.draft_k >= 0 ? std::min({o.draft_k, dd.n, cap})
                                                     : sched.choose(dd.accept.data(), dd.n, cap);
                        std::vector<int> window{tok};
                        window.insert(window.end(), dd.tokens.begin(), dd.tokens.begin() + k);
                        const double v0 = now_ms();
                        // a request that samples: row t is sampled as the plain path would sample that token (the
                        // same chain, counter and penalty history), and a draft is kept only when it equals the pick
                        const VerifyResult v = e.verify(window, e.position(), !r.greedy());
                        const double verify_ms = now_ms() - v0;
                        std::vector<int> next = v.next;
                        const int keep = r.greedy() ? round_keep(window, v.next, o.eos)
                                                    : round_picks(window, [&](int t) {
                                                          return sampler->pick(v.logits[(size_t) t], r,
                                                                               row_history(s.live, window, t, r.penalty_last_n),
                                                                               oc.seed, oc.counter + (uint64_t) t);
                                                      }, o.eos, next);
                        if (!r.greedy()) oc.counter += (uint64_t) keep;
                        e.commit(keep);
                        s.live.insert(s.live.end(), window.begin(), window.begin() + keep);
                        add_timing();
                        sched.observe_round(k, verify_ms, now_ms() - r0, keep);
                        if (k == 0) sched.observe_step(verify_ms);   // a one-row window costs about a step
                        ++d_rounds;
                        d_drafted += k;
                        d_accepted += keep - 1;
                        for (int i = 0; i < keep && !ended; ++i) {
                            tok = next[(size_t) i];
                            ended = emit(tok);
                        }
                        continue;
                    }
                    greedy_next = e.step(tok, e.position());
                    s.live.push_back(tok);
                    add_timing();
                    if (o.dspark) sched.observe_step(e.last_timing().total_ms);
                    tok = r.greedy() ? greedy_next : sampler->pick(e.last_logits(), r, s.live, oc.seed, oc.counter++);
                    ended = emit(tok);
                }
            }
            d.decode_ms = now_ms() - t0;
            say("%s", serve::done_line(d).c_str());
            std::fprintf(stderr,
                         "ds41_serve: prompt %lld (reused %lld, read %lld) %.0f ms, %lld out %.1f ms/token, "
                         "VRAM hits %lld of %lld, %s; DSpark windows %lld, drafts %lld, accepted %lld\n",
                         n, d.reused, d.read, d.prompt_ms, d.generated,
                         d.generated > 1 ? d.decode_ms / (double) (d.generated - 1) : 0.0, d.hits, d.lookups,
                         d.finish.c_str(), d_rounds, d_drafted, d_accepted);
            std::fflush(stderr);
            oc.status = Outcome::kDone;
            oc.finish = d.finish;
            oc.generated = d.generated;
            oc.t0 = r0;
        } catch (const std::exception& ex) {
            e.set_prefill_progress(nullptr);
            say("ERR %s", ex.what());
            try {
                s.restart();   // the next request starts over at position 0
            } catch (const std::exception& again) {
                // the engine refuses every call after a failure that left its state half written: end, so the
                // server starts a new one
                std::fprintf(stderr, "ds41_serve: the engine is unusable (%s); exiting\n", again.what());
                oc.status = Outcome::kFatal;
            }
        }
        return oc;
    };

    // One decode step of every active slot (upstream's batch window): BT per slot, BDONE for a slot that ends.
    auto batch_step = [&]() -> bool {
        std::vector<int> rows, toks;
        for (int b = 0; b < nslots; ++b)
            if (slots[(size_t) b].active) {
                rows.push_back(b);
                toks.push_back(slots[(size_t) b].x);
            }
        try {
            const std::vector<int> next = e.step_slots(rows, toks);
            for (size_t i = 0; i < rows.size(); ++i) {
                Slot& sl = slots[(size_t) rows[i]];
                sl.ids.push_back(sl.x);
                const int tok = sl.r.greedy() ? next[i]
                                              : sampler->pick(e.slot_logits((int) i), sl.r, sl.ids, sl.seed, sl.counter++);
                say("BT %d %d", rows[i], tok);
                ++sl.produced;
                const char* fin = nullptr;
                if (std::find(o.eos.begin(), o.eos.end(), tok) != o.eos.end()) fin = "stop";
                else if (sl.stop) fin = "cancel";
                else if (sl.produced >= sl.max_new || (int) sl.ids.size() + 2 > o.max_context) fin = "length";
                if (fin) {
                    say("BDONE %d %lld %s %.1f", rows[i], sl.produced, fin, now_ms() - sl.t0);
                    sl.active = sl.stop = false;
                    sl.cached = true;   // its state holds sl.ids: a later turn of this conversation continues there
                } else {
                    sl.x = tok;
                }
            }
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "ds41_serve: a batch step failed (%s); exiting\n", ex.what());
            return false;
        }
        return true;
    };
    auto any_active = [&] {
        return std::any_of(slots.begin(), slots.end(), [](const Slot& sl) { return sl.active; });
    };

    std::string line;
    for (;;) {
        if (any_active()) {
            if (!in.poll(line)) {
                if (!batch_step()) return 1;
                continue;
            }
        } else {
            write_roots();   // idle: a new root goes to disk now, not inside a request
            if (!in.next(line)) break;
        }
        if (line == "QUIT") break;
        if (line.empty()) continue;
        if (line.rfind("BSTOP ", 0) == 0) {   // the slot's next window writes one more BT, then BDONE ... cancel
            const int b = std::atoi(line.c_str() + 6);
            if (b >= 0 && b < nslots && slots[(size_t) b].active) slots[(size_t) b].stop = true;
            continue;
        }
        if (line.rfind("BYIELD ", 0) == 0) {   // optional in upstream's protocol: a prompt here is read in one go
            std::fprintf(stderr, "ds41_serve: BYIELD ignored (a prompt is read without yielding)\n");
            continue;
        }
        in.stop.store(false);   // a STOP that arrived between requests is stale
        if (line == "VRAM" || line.rfind("VRAM ", 0) == 0) {
            say("ERR VRAM is not supported by ds41_serve");
            continue;
        }
        if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
            // BGEN <slot> <max_new> [keys] ids: the prompt and its first token in the main session (GEN 1), then the
            // state goes into the slot, which decodes the rest in the batch windows. BADM 1 | 0 always follows
            // (the server waits for it), also after an ERR.
            const bool image = line.rfind("BGENI ", 0) == 0;
            const char* p = line.c_str() + (image ? 6 : 5);
            char* end = nullptr;
            const long b = std::strtol(p, &end, 10);
            const long long mn = end && *end == ' ' ? std::strtoll(end + 1, &end, 10) : 0;
            const bool ok_slot = b >= 0 && b < nslots && !slots[(size_t) b].active;
            if (image || !ok_slot || mn < 1 || !end) {
                say("ERR %s", image ? "this engine has no image input" : "BGEN: no such free slot, or a bad max_new");
                say("BADM %ld 0", b);
                continue;
            }
            Request r;
            const std::string bad = serve::parse_gen("GEN 1" + std::string(end), r);
            if (!bad.empty()) {
                say("ERR bad request: %s", bad.c_str());
                say("BADM %ld 0", b);
                continue;
            }
            const Outcome oc = serve_gen(r);
            if (oc.status == Outcome::kFatal) return 1;
            const bool cont = oc.status == Outcome::kDone && oc.finish == "length" && oc.generated == 1 && mn > 1;
            if (cont) {
                try {
                    e.copy_to_slot((int) b);
                } catch (const std::exception& ex) {
                    std::fprintf(stderr, "ds41_serve: copy into slot %ld failed (%s); exiting\n", b, ex.what());
                    return 1;
                }
                Slot& sl = slots[(size_t) b];
                sl = Slot{};
                sl.active = true;
                sl.x = oc.last;
                sl.produced = 1;
                sl.max_new = mn;
                sl.r = r;
                sl.r.max_new = mn;
                sl.seed = oc.seed;
                sl.counter = oc.counter;
                sl.ids = s.live;
                sl.t0 = oc.t0;
            }
            say("BADM %ld %d", b, cont ? 1 : 0);
            continue;
        }
        if (line.rfind("GENI ", 0) == 0) {
            say("ERR this engine has no image input");
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) {
            say("ERR expected: GEN <max_new> <id,id,...> or BGEN <slot> <max_new> <id,id,...>");
            continue;
        }
        Request r;
        const std::string bad = serve::parse_gen(line, r);
        if (!bad.empty()) {
            say("ERR bad request: %s", bad.c_str());
            continue;
        }
        if (serve_gen(r).status == Outcome::kFatal) return 1;
    }
    return 0;
}
