// include/strata/ds41/host_experts.hpp - the RAM tier of the routed experts (upstream's resident budget,
// docs/DETAILS.md "A RAM budget", `--resident-budget-gib N`).
//
// The 3bpw experts (190.5 GiB) do not fit a 128 GB PC. The hottest experts that the VRAM tier does not hold are
// copied at start into one RAM arena of at most `budget` bytes, in profile rank order; the CPU expert kernel reads
// them there, or K10 reads their mapped device aliases. Every other expert stays in the mapped experts.bin.
// Those experts come through the OS file cache (the SSD tier).
// So the RAM tier holds the complement of the VRAM tier, as upstream's does.
//
// Experts may differ in size (SAGE 1.59bpw: 4.3 to 21.1 MiB). As upstream's resident complement
// (src/core/expert_source.cpp pin_cache_complement), the arena is compact: each slot is as large as the expert that
// first filled it, and the slots follow each other. An expert that does not fit the rest of the budget is skipped;
// smaller ones after it may still enter.
//
// When the adaptive VRAM tier swaps expert X in (from a RAM slot) and Y out, Y takes X's RAM slot: Y is copied from
// VRAM to a pinned staging buffer, X from its RAM slot to VRAM, then Y from staging to the RAM slot. Nothing is read
// from the SSD. While the copies run, X is computed from the file (its RAM slot is being overwritten). Y must fit the
// slot's capacity (the VRAM tier only plans such swaps).
#pragma once

#include "strata/ds41/pack.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::ds41 {

namespace kernels { struct Exl3Expert; }

/// The experts the RAM tier holds: the profile's pairs in rank order, skipping the ones the VRAM tier holds
/// (vram_res >= 0) and the ones that do not fit the rest of `budget`. vram_res and bytes: [n_layers][n_experts].
std::vector<std::pair<int, int>> plan_ram_tier(const std::vector<std::pair<int, int>>& ranked,
                                               const std::vector<int32_t>& vram_res, int n_experts,
                                               const std::vector<uint64_t>& bytes, uint64_t budget);

/// RAM for the expert arena when no budget is given: the smaller of MemAvailable and the container's limit (cgroup v2
/// memory.max or v1 memory.limit_in_bytes) less its anonymous memory, less `headroom` (upstream keeps 4 GB free).
/// Never negative.
size_t auto_ram_budget(size_t headroom);

class HostExperts {
public:
    /// Copies the planned experts into the arena (`threads` readers), points the CPU kernel's layers (cpu_handles,
    /// one per layer; none: no CPU kernel, for tests) at the copies, and hands their file pages back to the OS.
    /// budget 0: no RAM tier.
    HostExperts(const Pack& pack, const std::vector<std::pair<int, int>>& ranked, const std::vector<int32_t>& vram_res,
                size_t budget, const std::vector<int64_t>& cpu_handles, int threads);
    ~HostExperts();
    HostExperts(const HostExperts&) = delete;
    HostExperts& operator=(const HostExperts&) = delete;

    int slots() const { return slots_; }
    /// the arena: the slots' capacities summed
    size_t arena_bytes() const { return arena_bytes_; }
    /// the largest slot
    size_t max_slot_bytes() const { return max_slot_bytes_; }
    bool locked() const { return locked_; }
    /// the slots were read with O_DIRECT (else copied from the mapped pack, through the file cache)
    bool filled_direct() const { return filled_direct_; }
    /// [n_layers][n_experts] on the device. Null when mapping is unavailable.
    /// An entry with w1.trellis == nullptr is CPU-only. The address stays fixed.
    const kernels::Exl3Expert* experts_dev() const { return experts_dev_; }
    /// RAM slot of (layer, expert), or -1 (the expert is read from the file)
    int32_t slot_of(int layer, int expert) const { return slot_[(size_t) layer * n_experts_ + expert]; }
    uint8_t* slot_ptr(int slot) const { return arena_ + off_[slot]; }
    /// the largest expert the slot can hold: the size of the expert that first filled it
    size_t slot_capacity(int slot) const { return off_[slot + 1] - off_[slot]; }

    /// Point the CPU kernel's (layer, expert) at its bytes in the file (the mapped experts.bin). It leaves the RAM
    /// tier's table; its slot stays reserved for the next assign.
    /// Call only between steps. This also revokes the device descriptor before a slot is overwritten.
    void point_to_file(int layer, int expert);
    /// Record that `slot` now holds (layer, expert) and point the CPU kernel at it. The slot's previous expert
    /// must already point elsewhere (point_to_file) and leaves the RAM tier. Throws if the expert is larger than
    /// the slot's capacity.
    /// Call only between steps, after the slot copy has completed.
    void assign(int slot, int layer, int expert);

private:
    void point(int layer, int expert, const uint8_t* bytes);
    void publish_descriptor(int layer, int expert, int slot);

    const Pack& pack_;
    std::vector<int64_t> handles_;
    int n_experts_;
    int slots_ = 0;
    std::vector<size_t> off_;                       ///< [slots + 1] slot offsets in the arena
    size_t max_slot_bytes_ = 0;
    uint8_t* arena_ = nullptr;
    size_t arena_bytes_ = 0;
    bool locked_ = false;
    bool filled_direct_ = false;
    bool registered_ = false;
    uint8_t* device_alias_ = nullptr;
    kernels::Exl3Expert* experts_dev_ = nullptr;
    std::vector<int32_t> slot_;                     ///< [n_layers][n_experts] RAM slot or -1
    std::vector<std::pair<int, int>> holder_;       ///< [slots] (layer, expert) in each slot
};

}  // namespace strata::ds41
