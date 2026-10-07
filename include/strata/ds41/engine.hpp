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
#include <memory>
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
