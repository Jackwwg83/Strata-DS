// include/strata/ds41/engine.hpp - DeepSeek V4.1 Flash decode, one token at a time (M1: correctness first).
//
// GPU: everything except the routed experts. CPU: routed experts with exllamav3's moe_mul1, reading the
// mmap'ed experts.bin in place (upstream Strata's "CPU computes the misses in RAM", with no VRAM cache yet).
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

class Engine {
public:
    Engine(const std::string& pack_dir, int max_seq, int cpu_threads);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// Run token `token` at position `pos` (0, 1, 2, ... in order). Returns the greedy next token.
    /// With `dump` non-null, fills it for this step.
    int step(int token, int pos, StepDump* dump = nullptr);

    /// FP32 logits of the last step (all 129280)
    const std::vector<float>& last_logits() const;

    struct Timing { double gpu_ms = 0, cpu_experts_ms = 0, engram_ms = 0, total_ms = 0; };
    const Timing& last_timing() const { return timing_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Timing timing_;
};

}  // namespace strata::ds41
