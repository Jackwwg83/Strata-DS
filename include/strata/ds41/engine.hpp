// include/strata/ds41/engine.hpp - DeepSeek V4.1 Flash decode, one token at a time.
//
// The GPU computes dense work, VRAM experts, and a quota of experts from mapped RAM (DS41_ZC_QUOTA, default 4).
// The CPU computes the remaining RAM experts and all file experts with exllamav3's moe_mul1.
// An ExpertDoorbell joins both results per layer. K10 uses the same math for VRAM and mapped RAM.
// Token 0 runs the same path as every later token; the prototype's prefill of one token is equivalent.
#pragma once

#include "strata/ds41/pack.hpp"
#include "strata/ds41/verify.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::ds41 {

struct StepDump {
    std::vector<uint16_t> hidden;                    ///< [40][4][5120] bf16 bits: the hc stream after each block
    std::vector<std::array<int, 6>> routes;          ///< [40] routed expert ids
    std::vector<std::array<float, 6>> weights;       ///< [40] routing weights
    std::vector<std::pair<int, float>> top_logits;   ///< 8 best (token, logit)
};

struct EngineOptions {
    int max_seq = 4096;
    int cpu_threads = 8;
    std::string expert_profile;             ///< STRP profile for the VRAM expert tier; empty: no tier
    int64_t vram_expert_slots = -1;         ///< with a profile: -1 = as many as fit, 0 = none
    size_t vram_reserve_bytes = 1536ull << 20;   ///< VRAM left free when the slot count is automatic
    int adapt_every = 4;                    ///< adaptive tier: steps between swaps (0 = static residency)
    float adapt_decay = 0.7f;
    int adapt_swaps = 96;
    /// with a profile: RAM tier size in GiB (upstream's resident budget); -1 = automatic, the available RAM less
    /// 24 GiB (upstream setup's default N = RAM - 24 GB: N = 40 on a 64 GB PC); 0 = none. Measured on an RTX 4090
    /// with 119.9 GiB of container RAM and an 8.8 GB/s SSD (2026-10-06, 3bpw): no tier, 8K prompt 318 tok/s and
    /// decode 168-250 ms/token; 96 GiB, 582 tok/s and 103-109 ms/token (32K: 966 -> 1,031 tok/s).
    double ram_budget_gib = -1;
    /// Batched prefill (M3), layer-major: tokens per pass at most (halved until the scratch fits); a pass copies every
    /// expert to the GPU once, so one pass for the whole prompt is the fastest. 0 = prefill() runs step() token by
    /// token. Inside a layer the pass runs in sub-batches of prefill_batch tokens. Scratch and the expert ring come
    /// from VRAM tier slots lent for the call (upstream), else from cudaMalloc.
    int prefill_chunk = 65536;
    int prefill_batch = 4096;   ///< sub-batch tokens (halved down to 512 when the scratch does not fit)
    int prefill_ring = 256;     ///< expert ring slots at most (halved down to 16 when they do not fit)
    /// snapshot slots (save_snapshot): the sliding-window rings and compressor states of all layers, ~5.3 MB of VRAM
    /// each. ds41_serve keeps one before each prompt's last token, so a chat whose next prompt changes the last
    /// answer's start (DeepSeek drops earlier reasoning) goes back there instead of reading everything again.
    int snapshots = 4;
    /// concurrent requests (ds41_serve --batch, upstream's batch slots): slots with their own attention state that
    /// decode together, at most kVerifyMaxTokens (the CPU expert kernel's rows). 0: none. Each slot holds about
    /// 90 MB of VRAM at 32K context, and the slots share a staging area of 4 x 6 experts.
    int batch_slots = 0;
    int prefill_threads = 16;       ///< expert stream readers (pread from the pack)
    int prefill_host_buffers = 64;  ///< pinned staging buffers of the expert stream (how far reads run ahead)
};

/// What one prefill() call did
struct PrefillTiming {
    /// engram_ms: the GPU side waited for engram rows (read on their own thread); stream_wait: for expert copies
    double total_ms = 0, engram_ms = 0, stream_wait_ms = 0;
    int64_t engram_rows = 0, engram_unique = 0;                ///< engram rows used, and distinct rows read
    int chunks = 0, chunk_tokens = 0, sub_batch = 0;           ///< passes, tokens per pass, tokens per sub-batch
    int64_t vram_experts = 0;                                  ///< (layer, expert) pairs computed from VRAM slots
    int64_t streamed = 0, from_ram = 0, from_cache = 0, from_ssd = 0;   ///< pairs copied through the ring, by source
};

/// Thrown by prefill() when the progress callback returned false. The engine is then back at position 0.
struct PrefillCancelled : std::runtime_error {
    PrefillCancelled() : std::runtime_error("ds41 prefill: cancelled") {}
};

/// prefill() progress: tokens of the call done and the call's total. Inside a pass, which runs layer by layer, done
/// is the share of the layers enqueued. Return false to stop the prefill.
using PrefillProgress = std::function<bool(int done, int total)>;

class Engine {
public:
    Engine(const std::string& pack_dir, const EngineOptions& opt);
    Engine(const std::string& pack_dir, int max_seq, int cpu_threads);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// Run token `token` at position `pos` (0, 1, 2, ... in order). Returns the greedy next token.
    /// With `dump` non-null, fills it for this step.
    int step(int token, int pos, StepDump* dump = nullptr);

    /// Check a tentative window at the committed position. Call commit before step, prefill, or verify again.
    /// Each output is the greedy next token after that input row. CPU windows are capped at four rows.
    VerifyResult verify(const std::vector<int>& window, int pos, bool logits = false);
    /// Keep the first n_keep inputs (1..T). Discard every later input and its state.
    void commit(int n_keep);

    /// Feed tokens at positions pos, pos + 1, ... (pos = the tokens fed so far) in batched chunks: every layer on the
    /// GPU, all routed experts on the GPU (VRAM tier slots, the rest streamed through a ring). Returns the greedy
    /// token after the last one; last_logits() holds its logits. Decode continues with step(next, pos + n).
    /// nll non-null: (*nll)[i] = -log p(tokens[i + 1] | tokens[0..i]) for i < n - 1.
    int prefill(const std::vector<int>& tokens, int pos, std::vector<float>* nll = nullptr);
    const PrefillTiming& last_prefill() const { return prefill_timing_; }
    /// Called after every layer of a prefill pass (token by token without passes); null: none. A false return makes
    /// prefill() stop where it is, reset the engine and throw PrefillCancelled.
    void set_prefill_progress(PrefillProgress fn);

    /// Forget every token fed so far: the next call feeds position 0. A pending verify window is dropped. The caches
    /// need no clearing: each row is written before it is read, and the lengths come from the position.
    void reset();
    /// Tokens fed so far: the position of the next step or prefill
    int position() const;

    /// Save the state at the current position in `slot` (0 .. snapshot_slots() - 1). Only the state that the position
    /// does not determine is copied (the window rings and the compressor states); the compressed rows are written
    /// before they are read, so they need no copy.
    void save_snapshot(int slot);
    /// Go back to the position saved in `slot`; returns it. The caller makes sure that the tokens fed before that
    /// position are still the ones fed when it was saved. Refused when the slot is empty or past position(). A pending
    /// verify window is dropped. The slot stays valid.
    int restore_snapshot(int slot);
    int snapshot_slots() const;

    /// Concurrent requests. A request is read in the main session (prefill, step, verify), then copied into a slot.
    int batch_slots() const;
    /// The main session's state and tokens -> slot (what the slot held is gone)
    void copy_to_slot(int slot);
    /// slot -> the main session (a later turn of the slot's conversation continues from there)
    void copy_from_slot(int slot);
    /// Tokens fed to a slot so far: its next position
    int slot_position(int slot) const;
    /// One decode step of several slots (distinct, 1..kVerifyMaxTokens of them): row i feeds tokens[i] at slot
    /// slots[i]'s position. The dense weights and the experts are read once for all rows; each row attends over its
    /// own slot. Returns the greedy next token of each row; slot_logits(i) holds row i's FP32 logits until the next
    /// call. With static residency (adapt_every 0) a row's tokens equal those of the same sequence decoded alone.
    std::vector<int> step_slots(const std::vector<int>& slots, const std::vector<int>& tokens);
    const float* slot_logits(int row) const;

    /// FP32 logits of the last step (all 129280)
    const std::vector<float>& last_logits() const;

    /// engram_ms: the engram reads at the step start. gpu_ms: the rest of the step (wall time). cpu_experts_ms: the
    /// time the CPU thread spent computing experts, inside gpu_ms.
    /// expert_hits: routed experts of the step computed from VRAM slots, of expert_total.
    struct Timing {
        double gpu_ms = 0, cpu_experts_ms = 0, engram_ms = 0, total_ms = 0;
        int expert_hits = 0, expert_total = 0;   ///< VRAM hits and all routed uses
        int cpu_experts() const { return ram_experts + file_experts; }
        int zero_copy_experts() const { return expert_total - expert_hits - cpu_experts(); }
        int vram_swaps = 0;   ///< adaptive swaps committed before this step
        /// the CPU's experts by tier: from the RAM copy, from the mapped file, and of those, the ones with pages
        /// missing from RAM when computed (read from the SSD)
        int ram_experts = 0, file_experts = 0, ssd_experts = 0;
        int warmed = 0, warmed_useful = 0;   ///< lookahead: file-tier experts warmed, and of those, used next layer
        int prefetched = 0;   ///< misses the GPU computed from the prefetch buffer (DS41_PREFETCH), in zero_copy_experts()
    };
    /// VRAM expert slots in use (0: no tier)
    int vram_expert_slots() const;
    const Timing& last_timing() const { return timing_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Timing timing_;
    PrefillTiming prefill_timing_;
};

}  // namespace strata::ds41
