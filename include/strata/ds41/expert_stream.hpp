// include/strata/ds41/expert_stream.hpp - prefill's expert stream (upstream's Stager and ring).
//
// Prefill computes every routed expert on the GPU. Experts that are not in the VRAM tier are copied into a ring of
// device slots while the GPU computes earlier ones. Reader threads take the jobs in order; job j uses ring slot
// j % slots and starts once the consumer has released job j - slots (an event on the consumer's stream). A reader
// copies an expert from its RAM-tier slot (pinned) or reads it with pread into its own pinned buffer, then copies it
// to the device on its own stream.
//
// File cache: an expert whose pages were all cached before the read stays cached. An expert read (at least in part)
// from the SSD is dropped from the cache after the copy (posix_fadvise DONTNEED): one prefill streams the whole
// 190 GB file, and keeping it would push out the experts that decode reads from the cache.
#pragma once

#include "strata/ds41/pack.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::ds41 {

class HostExperts;

class ExpertStream {
public:
    /// ring: `slots` device slots of `slot_bytes` each. host: the RAM tier or null. threads: readers (one pinned
    /// buffer of slot_bytes each).
    ExpertStream(const Pack& pack, const HostExperts* host, uint8_t* ring, int slots, size_t slot_bytes, int threads);
    ~ExpertStream();
    ExpertStream(const ExpertStream&) = delete;
    ExpertStream& operator=(const ExpertStream&) = delete;

    /// Append jobs (layer, expert) in the order the consumer will take them. Returns the index of the first.
    int64_t push(const std::vector<std::pair<int, int>>& jobs);
    /// Blocks until job j's bytes are in its slot; returns the slot. Throws if a read or copy failed.
    uint8_t* wait(int64_t j);
    /// The consumer is done with job j once the work enqueued on `stream` so far has run: job j + slots may then
    /// overwrite the slot. Release jobs in order.
    void release(int64_t j, cudaStream_t stream);
    /// Blocks until every pushed job is copied and released, then forgets the jobs (job numbers restart at 0).
    void drain();

    struct Stats {
        int64_t jobs = 0, from_ram = 0, from_cache = 0, from_ssd = 0;   ///< experts by source
        double consumer_wait_ms = 0;                                     ///< time wait() blocked
    };
    Stats take_stats();
    int slots() const { return slots_; }

private:
    void reader(int id);

    const Pack& pack_;
    const HostExperts* host_;
    uint8_t* ring_;
    int slots_;
    size_t slot_bytes_;
    int fd_ = -1;
    int device_ = 0;
    std::vector<std::thread> threads_;
    std::vector<cudaEvent_t> used_;            ///< per slot: recorded by release()

    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::string error_;
    std::vector<std::pair<int, int>> jobs_;
    std::vector<uint8_t> done_;                 ///< per job: copied
    int64_t next_ = 0;                          ///< next job a reader takes
    int64_t released_ = 0;                      ///< jobs [0, released_) are released
    std::atomic<int64_t> n_ram_{0}, n_cache_{0}, n_ssd_{0};
    double wait_ms_ = 0;
};

}  // namespace strata::ds41
