// Strata-DS K11-04 patch. Included after the unchanged vendor assign_gemvs helper.
// Dynamic tickets adapt to different worker speeds without moving any arithmetic,
// changing requested thread counts, or changing the pool's phase barriers.
#pragma once

inline bool strata_dynamic_schedule_enabled(const ForwardCtx& c)
{
    // Retain the original large-prefill cache/reuse policy and every AVX-512 tier.
    return c.m_total >= 1 && c.m_total <= 8 &&
           (g_isa == Isa::Avx2 || g_isa == Isa::AvxVnni);
}

template <typename Gemv>
inline void strata_assign_gemvs(ForwardCtx& c, int worker, int num_workers,
                               int total, int tiles_n, Gemv gemv)
{
    if (!strata_dynamic_schedule_enabled(c) || num_workers <= 1 ||
        tiles_n <= 0 || tiles_n % 8 != 0)
    {
        assign_gemvs(worker, num_workers, total, tiles_n, gemv);
        return;
    }

    // A whole 8-tile/128-column band is indivisible. For positive int dimensions,
    // multiplication in int64_t cannot overflow. Each ticket belongs to one matrix
    // and one complete band, including when a prepared token uses two quant rows.
    const int64_t bands = tiles_n / 8;
    const int64_t count = static_cast<int64_t>(total) * bands;
    for (;;)
    {
        const int64_t ticket = c.next_gemv_band.fetch_add(1, std::memory_order_relaxed);
        if (ticket >= count) return;
        const int j = static_cast<int>(ticket / bands);
        const int t0 = static_cast<int>(ticket % bands) * 8;
        gemv(j, t0, t0 + 8);
    }
}
