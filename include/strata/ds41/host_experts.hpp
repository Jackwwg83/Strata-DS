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
//
// The adaptive tier (DS41_RAM_ADAPT=N, ds41/docs/cache-design-2026-10-08.html): the static tier above holds the
// profile's experts for good, and every other expert comes through the OS file cache, page fault by page fault. With
// N > 0 the tier follows the conversation instead. N slots of each capacity stay free. An expert that is in no tier
// when the CPU needs it is read from the SSD (O_DIRECT) into a free slot and computed there; it stays. Between steps
// the new experts are published (the GPU may then read them too), and the least recently used experts leave until
// each capacity has N free slots again. During a step only free slots are written, so no reader sees a slot change.
// On the laptop's recorded conversations this cut the SSD reads per token from 17.5 to 6.5 (tools/ds41/cache_sim.py).
#pragma once

#include "strata/ds41/pack.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata::ds41 {

namespace kernels { struct Exl3Expert; }
class ThreadPool;

namespace detail {
/// tests: the next HostExperts::admit() read throws once
inline std::atomic<bool>& admit_fault() {
    static std::atomic<bool> f{false};
    return f;
}
/// tests: pinning the arena fails while it is larger than this many bytes (0: never)
inline std::atomic<size_t>& register_limit() {
    static std::atomic<size_t> b{0};
    return b;
}
/// tests: the size of admit()'s read parts (a multiple of 4 KiB; 1 MiB in use)
inline std::atomic<size_t>& admit_part_bytes() {
    static std::atomic<size_t> b{1u << 20};
    return b;
}
}  // namespace detail

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
    /// the GPU can read the arena (pinned and mapped)
    bool mapped() const { return device_alias_ != nullptr; }
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
    /// Point the CPU kernel's (layer, expert) at bytes held elsewhere in host memory (an expert on its way from a
    /// VRAM slot to a RAM slot, held in a swap buffer); held() is true until assign() or point_to_file(). The device
    /// descriptor is revoked: the GPU does not read it from there. Call only between steps.
    void point_to(int layer, int expert, const uint8_t* bytes);
    /// the CPU reads (layer, expert) from host memory: its RAM slot or a swap buffer (not the file)
    bool in_memory(int layer, int expert) const {
        const size_t i = (size_t) layer * n_experts_ + expert;
        return slot_[i] >= 0 || (!held_.empty() && held_[i]);
    }

    // ---- the adaptive tier ----
    /// Keep `reserve` slots of each capacity free: per capacity, the lowest-ranked experts leave the tier (to the
    /// file) until that many slots are free. Call once, between steps, with no VRAM swap in flight (it throws
    /// otherwise). 0: the static tier.
    void enable_adapt(int reserve);
    int reserve() const { return reserve_; }
    /// admit() reads with O_DIRECT (each read goes to the SSD); false: it copies from the mapped pack
    bool reads_direct() const { return dfd_ >= 0; }
    /// During a step, on the CPU worker: read ids[0..n) of `layer`, which are in no tier, into free slots (for each
    /// the smallest capacity that holds it) and point the CPU kernel at them. ok[i] is false when no free slot holds
    /// expert i or its read failed: the CPU kernel then still reads it from the file. The RAM table and the device
    /// descriptors do not change until end_step(). Returns the number read.
    int admit(int layer, const int32_t* ids, int n, bool* ok);
    /// Between steps: publish the experts admitted during the step, record the step's routes ([n_layers][topk],
    /// negative ids ignored) as uses, then free the least recently used slots until each capacity has `reserve` free
    /// slots. A locked slot is never freed. Returns the number of experts that left.
    int end_step(const int32_t* routes, int topk);
    /// A VRAM swap reads or writes `slot` until unlock(): end_step() does not free it.
    void lock(int slot);
    void unlock(int slot);
    /// (layer, expert) entered VRAM: its RAM slot, if it has one, becomes free. Between steps.
    void release(int layer, int expert);
    /// free slots (all capacities)
    int free_slots() const;
    int64_t admitted_total() const { return admitted_total_; }
    int64_t evicted_total() const { return evicted_total_; }

private:
    void point(int layer, int expert, const uint8_t* bytes);
    void free_slot(int slot);      ///< its expert (if any) leaves for the file; the slot joins the free ones
    /// admit()'s reads into the picked slots (pick[i] < 0: none); records and points the ones read
    int read_picked(int layer, const int32_t* ids, int n, const std::vector<int>& pick, bool* ok);
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
    std::vector<uint8_t> held_;   ///< [n_layers][n_experts] pointed at a swap buffer (point_to)
    bool registered_ = false;
    uint8_t* device_alias_ = nullptr;
    kernels::Exl3Expert* experts_dev_ = nullptr;
    std::vector<int32_t> slot_;                     ///< [n_layers][n_experts] RAM slot or -1
    std::vector<std::pair<int, int>> holder_;       ///< [slots] (layer, expert) in each slot; (-1, -1): free
    // adaptive tier
    int reserve_ = 0;
    int dfd_ = -1;                                  ///< experts.bin opened with O_DIRECT (-1: copy from the map)
    std::unique_ptr<ThreadPool> readers_;           ///< admit()'s reader threads, started with the adaptive tier
    std::vector<uint8_t> free_, busy_;              ///< [slots] free; locked by a VRAM swap
    std::vector<uint64_t> age_;                     ///< [slots] last use
    uint64_t clock_ = 0;
    std::vector<std::vector<int>> classes_;         ///< slots by capacity, smallest capacity first
    std::mutex admit_mu_;
    std::vector<std::pair<int, std::pair<int, int>>> admitted_;   ///< (slot, (layer, expert)) read this step
    int64_t admitted_total_ = 0, evicted_total_ = 0;
};

}  // namespace strata::ds41
