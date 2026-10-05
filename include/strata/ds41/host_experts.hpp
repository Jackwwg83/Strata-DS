// include/strata/ds41/host_experts.hpp - the RAM tier of the routed experts (upstream's resident budget,
// docs/DETAILS.md "A RAM budget", `--resident-budget-gib N`).
//
// The 3bpw experts (190.5 GiB) do not fit a 128 GB PC. The hottest experts that the VRAM tier does not hold are
// copied at start into one RAM arena of `budget` bytes, in profile rank order; the CPU expert kernel reads them
// there. Every other expert stays in the mapped experts.bin and comes through the OS file cache (the SSD tier).
// So the RAM tier holds the complement of the VRAM tier, as upstream's does.
//
// When the adaptive VRAM tier swaps expert X in (from a RAM slot) and Y out, Y takes X's RAM slot: Y is copied from
// VRAM to a pinned staging buffer, X from its RAM slot to VRAM, then Y from staging to the RAM slot. Nothing is read
// from the SSD. While the copies run, X is computed from the file (its RAM slot is being overwritten).
#pragma once

#include "strata/ds41/pack.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::ds41 {

/// The experts the RAM tier holds: the profile's pairs in rank order, skipping the ones the VRAM tier holds
/// (vram_res >= 0), at most n_slots. vram_res: [n_layers][n_experts].
std::vector<std::pair<int, int>> plan_ram_tier(const std::vector<std::pair<int, int>>& ranked,
                                               const std::vector<int32_t>& vram_res, int n_experts, int64_t n_slots);

/// RAM for the expert arena when no budget is given: the smaller of MemAvailable and the container's limit less its
/// anonymous memory, less `headroom` (upstream keeps 4 GB free). Never negative.
size_t auto_ram_budget(size_t headroom);

class HostExperts {
public:
    /// Copies the planned experts into the arena (`threads` readers), points the CPU kernel's layers (cpu_handles,
    /// one per layer) at the copies, and hands their file pages back to the OS. budget 0: no RAM tier.
    HostExperts(const Pack& pack, const std::vector<std::pair<int, int>>& ranked, const std::vector<int32_t>& vram_res,
                size_t budget, const std::vector<int64_t>& cpu_handles, int threads);
    ~HostExperts();
    HostExperts(const HostExperts&) = delete;
    HostExperts& operator=(const HostExperts&) = delete;

    int slots() const { return slots_; }
    size_t slot_bytes() const { return slot_bytes_; }
    bool locked() const { return locked_; }
    /// RAM slot of (layer, expert), or -1 (the expert is read from the file)
    int32_t slot_of(int layer, int expert) const { return slot_[(size_t) layer * n_experts_ + expert]; }
    uint8_t* slot_ptr(int slot) const { return arena_ + (size_t) slot * slot_bytes_; }

    /// Point the CPU kernel's (layer, expert) at its bytes in the file (the mapped experts.bin).
    void point_to_file(int layer, int expert);
    /// Record that `slot` now holds (layer, expert) and point the CPU kernel at it. The slot's previous expert
    /// must already point elsewhere (point_to_file) and leaves the RAM tier.
    void assign(int slot, int layer, int expert);

private:
    void point(int layer, int expert, const uint8_t* bytes);

    const Pack& pack_;
    std::vector<int64_t> handles_;
    int n_experts_;
    int slots_ = 0;
    size_t slot_bytes_ = 0;
    uint8_t* arena_ = nullptr;
    size_t arena_bytes_ = 0;
    bool locked_ = false;
    bool registered_ = false;
    std::vector<int32_t> slot_;                     ///< [n_layers][n_experts] RAM slot or -1
    std::vector<std::pair<int, int>> holder_;       ///< [slots] (layer, expert) in each slot
};

}  // namespace strata::ds41
