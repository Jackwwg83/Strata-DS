// include/strata/ds41/vram_experts.hpp - the VRAM expert tier (upstream include/strata/core/expert_cache.hpp).
//
// Some routed experts live in VRAM slots and the GPU computes them (task K10); the CPU computes only the misses.
// The slots are filled at start from a profile (tools/ds41/make_profile.py, upstream's STRP format) in rank order.
// The residency table maps (layer, expert) to a slot or -1; the doorbell publish reads it on the device, so the
// GPU and the CPU split every routed expert the same way.
//
// The adaptive tier (upstream src/program/generate.cpp, adapt()) makes the slots follow the conversation: decayed
// routing counts per (layer, expert); every few steps, per layer, the most-routed missing experts replace the
// least-routed resident ones when they were routed clearly more often. An evicted expert leaves the table at once
// (the CPU computes it); the new one enters once its copy has landed. The copies run on their own thread and
// stream, because the pack is mmap'ed (pageable) and a pageable copy holds the calling thread.
#pragma once

#include "strata/ds41/kernels/k10_exl3_moe.hpp"
#include "strata/ds41/pack.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::ds41 {

/// Reads a STRP profile: the ranked (layer, expert) pairs. Throws when the file is not a profile of this shape.
std::vector<std::pair<int, int>> read_expert_profile(const std::string& path, int n_layers, int n_experts);

struct ExpertSwap {
    float gain;
    int32_t layer, in, out;   ///< `in` (missing) takes the slot of `out` (resident), same layer
};
/// Upstream's swap choice. usage and res: [n_layers][n_experts]. Per layer, candidates are missing experts with
/// usage >= 2, victims the resident ones; the i-th most-used candidate pairs with the i-th least-used victim while
/// it leads by 1.5 or more. All layers' swaps, largest gain first, at most max_swaps.
std::vector<ExpertSwap> plan_expert_swaps(const std::vector<float>& usage, const std::vector<int32_t>& res,
                                          int n_layers, int n_experts, int max_swaps);

class VramExperts {
public:
    /// Workspace the K10 kernel gets (its interface asks for at least 64 MiB)
    static constexpr size_t kWorkspaceBytes = 64ull << 20;

    struct Adapt {
        int every = 4;          ///< steps between adaptations; 0 = static residency
        float decay = 0.7f;     ///< usage counts are multiplied by this after each adaptation
        int max_swaps = 96;
    };

    /// Fills `n_slots` slots with the profile's first pairs, copied from the mapped pack. n_slots < 0: as many as
    /// fit in the free VRAM after keeping `reserve_bytes` free (and the workspace). 0 slots is valid (no tier).
    VramExperts(const Pack& pack, const std::string& profile_path, int64_t n_slots, size_t reserve_bytes,
                Adapt adapt);
    ~VramExperts();
    VramExperts(const VramExperts&) = delete;
    VramExperts& operator=(const VramExperts&) = delete;

    int slots() const { return slots_; }
    size_t slot_bytes() const { return slot_bytes_; }
    /// [n_layers][n_experts] slot or -1, on the device; layer l starts at res_dev() + l * n_experts
    const int32_t* res_dev() const { return res_dev_; }
    const std::vector<int32_t>& res_host() const { return res_host_; }
    /// [slots] expert descriptors on the device (pointers into the slots)
    const kernels::Exl3Expert* experts_dev() const { return experts_dev_; }
    void* workspace() const { return ws_; }

    /// Count one step's routing: routes [n_layers][topk], host memory.
    void count(const int32_t* routes, int topk);
    /// Call between steps, with no work of the tier in flight on the device. Commits swaps whose copies have
    /// landed; every `every` calls plans new swaps, evicts their victims and starts the copies. Returns the
    /// number of swaps committed now.
    int between_steps();
    int64_t swaps_total() const { return swaps_total_; }

private:
    kernels::Exl3Expert describe(int layer, int expert, int slot) const;
    void upload_res();
    void copy_worker(std::vector<ExpertSwap> swaps, std::vector<int32_t> slots);

    const Pack& pack_;
    Adapt adapt_;
    int device_ = 0;
    int slots_ = 0;
    size_t slot_bytes_ = 0;
    uint8_t* arena_ = nullptr;
    int32_t* res_dev_ = nullptr;
    std::vector<int32_t> res_host_;
    kernels::Exl3Expert* experts_dev_ = nullptr;
    std::vector<kernels::Exl3Expert> desc_host_;
    void* ws_ = nullptr;
    // adaptive tier
    std::vector<float> usage_;
    int64_t calls_ = 0, swaps_total_ = 0;
    cudaStream_t copy_stream_ = nullptr;
    std::thread copier_;
    std::atomic<bool> copies_done_{false};
    bool copy_error_ = false;
    std::vector<std::pair<int32_t, int32_t>> pending_;   ///< (layer * n_experts + expert, slot) in flight
};

}  // namespace strata::ds41
