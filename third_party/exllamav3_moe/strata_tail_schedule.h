// Strata-DS K11-05: retain static homes; idle workers may claim unowned tails.
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>

namespace strata_moe {

// One packed interval is essential: independent head/tail atomics can let an
// owner and thief both claim the last band. Offsets are relative to a home.
struct alignas(64) TailRange {
    std::atomic<uint64_t> remaining;
};
constexpr uint32_t TAIL_OWNER_BANDS = 8;
constexpr uint32_t TAIL_STEAL_BANDS = 4;

inline bool tail_eligible(int rows, int workers, int total, int tiles_n)
{
    // Preserve the original whole-GEMV strided policy for many jobs. With this
    // bound a home has <= 4 * (INT_MAX / 8) bands, so packed uint32 offsets fit.
    return rows > 0 && rows <= 8 && workers > 1 && total > 0 &&
           static_cast<int64_t>(total) <= 4LL * workers &&
           tiles_n > 0 && tiles_n % 8 == 0;
}

inline int64_t tail_home_start(int worker, int workers, int64_t groups)
{
    return (groups / workers) * worker + (groups % workers) * worker / workers;
}

inline void tail_reset(TailRange* ranges, int workers, int total, int tiles_n)
{
    const int64_t groups = static_cast<int64_t>(total) * (tiles_n / 8);
    for (int w = 0; w < workers; ++w) {
        const uint32_t count = static_cast<uint32_t>(
            tail_home_start(w + 1, workers, groups) - tail_home_start(w, workers, groups));
        // Reserve each owner's first chunk before publication. Even a delayed
        // owner starts at exactly its original matrix/band, never a global queue.
        const uint32_t first = std::min(count, TAIL_OWNER_BANDS);
        ranges[w].remaining.store((uint64_t(count) << 32) | first, std::memory_order_relaxed);
    }
}

inline bool tail_claim(TailRange& range, bool thief, uint32_t& begin, uint32_t& end)
{
    uint64_t before = range.remaining.load(std::memory_order_relaxed);
    for (;;) {
        const uint32_t lo = static_cast<uint32_t>(before);
        const uint32_t hi = static_cast<uint32_t>(before >> 32);
        if (lo == hi) return false;
        const uint32_t size = std::min(hi - lo, thief ? TAIL_STEAL_BANDS : TAIL_OWNER_BANDS);
        const uint32_t next_lo = thief ? lo : lo + size;
        const uint32_t next_hi = thief ? hi - size : hi;
        const uint64_t after = (uint64_t(next_hi) << 32) | next_lo;
        if (range.remaining.compare_exchange_strong(before, after,
                std::memory_order_relaxed, std::memory_order_relaxed)) {
            begin = thief ? next_hi : lo;
            end = begin + size;
            return true;
        }
    }
}

template<class Gemv>
inline void tail_emit(int64_t begin, int64_t end, int bands_per_matrix, Gemv& gemv)
{
    while (begin < end) {
        const int matrix = static_cast<int>(begin / bands_per_matrix);
        const int band = static_cast<int>(begin % bands_per_matrix);
        const int count = static_cast<int>(std::min<int64_t>(end - begin, bands_per_matrix - band));
        gemv(matrix, band * 8, (band + count) * 8);
        begin += count;
    }
}

template<class Gemv>
inline void tail_assign(TailRange* ranges, int worker, int workers, int total, int tiles_n, Gemv gemv)
{
    const int bands_per_matrix = tiles_n / 8;
    const int64_t groups = static_cast<int64_t>(total) * bands_per_matrix;
    const int64_t home = tail_home_start(worker, workers, groups);
    const int64_t home_end = tail_home_start(worker + 1, workers, groups);
    tail_emit(home, std::min(home_end, home + TAIL_OWNER_BANDS), bands_per_matrix, gemv);
    uint32_t begin, end;
    while (tail_claim(ranges[worker], false, begin, end))
        tail_emit(home + begin, home + end, bands_per_matrix, gemv);
    // Bounded, one-pass tail recovery. No global ticket frontier, retry scan,
    // worker-count change, or attempt to reclaim already in-flight work.
    for (int step = 1; step < workers; ++step) {
        const int victim = (static_cast<int64_t>(worker) + step) % workers;
        if (tail_claim(ranges[victim], true, begin, end)) {
            const int64_t start = tail_home_start(victim, workers, groups);
            tail_emit(start + begin, start + end, bands_per_matrix, gemv);
        }
    }
}
} // namespace strata_moe
