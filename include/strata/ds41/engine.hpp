// include/strata/ds41/engine.hpp - DeepSeek V4.1 Flash decode, one token at a time.
//
// GPU: everything except the routed experts. CPU: routed experts with exllamav3's moe_mul1, reading the
// mmap'ed experts.bin in place (upstream Strata's "CPU computes the misses in RAM", with no VRAM cache yet).
// A CPU thread serves the experts layer by layer through an ExpertDoorbell while the GPU runs the shared expert.
// With a VRAM expert tier (VramExperts, task K10) the GPU computes the resident experts and the CPU the misses.
// Token 0 runs the same path as every later token; the prototype's prefill of one token is equivalent.
#pragma once

#include "strata/ds41/pack.hpp"

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

    /// FP32 logits of the last step (all 129280)
    const std::vector<float>& last_logits() const;

    /// engram_ms: the engram reads at the step start. gpu_ms: the rest of the step (wall time). cpu_experts_ms: the
    /// time the CPU thread spent computing experts, inside gpu_ms.
    /// expert_hits: routed experts of the step computed from VRAM slots, of expert_total.
    struct Timing {
        double gpu_ms = 0, cpu_experts_ms = 0, engram_ms = 0, total_ms = 0;
        int expert_hits = 0, expert_total = 0;
        int vram_swaps = 0;   ///< adaptive swaps committed before this step
    };
    /// VRAM expert slots in use (0: no tier)
    int vram_expert_slots() const;
    const Timing& last_timing() const { return timing_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Timing timing_;
};

}  // namespace strata::ds41
