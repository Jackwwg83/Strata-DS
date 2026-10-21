// include/strata/ds41/draft_scheduler.hpp - DSpark's policy, shared by ds41_generate and ds41_serve: when to draft
// and how many drafts a verify window checks (DeepSeek's scheduler and upstream's DraftPolicy: expected tokens per
// measured millisecond). Host only; the engine supplies the drafts' confidences and the measured times.
#pragma once

#include <algorithm>
#include <vector>

namespace strata::ds41 {

/// Before a round: draft only while drafting rounds commit more tokens per ms than plain steps (EMAs); after a losing
/// round, plain steps for a backoff (4, doubling to 64 while drafting keeps losing), then one drafting round again.
/// The first 16 tokens are plain steps.
/// After a draft (its time is spent): the k drafts with the most E(k) / verify(k), E(k) = 1 + S1 + .. + Sk, Si the
/// product of the first i confidence probabilities. Verify times are EMAs per k; a k not measured yet is the plain
/// step's time times 1 + 0.45 k (measured on the laptop: each verify row adds ~45% of a step).
struct DraftScheduler {
    static constexpr int kMax = 8;
    double vms[kMax] = {}, vn[kMax] = {};
    double step_ms = 0, spec_rate = 0;   // EMAs: a plain step's time, a drafting round's tokens per ms
    // the first 16 tokens are plain steps: they measure the step and keep drafting out of the cold start, when the
    // tiers are empty and every extra verify row reads its experts from the SSD (a 4-row window took 2.7 s)
    int backoff = 4, plain_left = 16;
    /// always (--draft-always): after the first 16 tokens every round drafts, k still chosen by choose(); no backoff.
    /// DGX Spark, EXL3 pack, SSD skip 2, ms/token code / zh / en: backoff 42.9 / 53.4 / 50.0, always 39.3 / 48.1 / 48.4
    /// (fixed k = 2: 39.8 / 49.7 / 48.6). A losing round there costs little: the RAM tier serves most verify rows.
    bool always = false;
    double verify_cost(int k) const {
        if (vn[k] > 0) return vms[k];
        const double base = step_ms > 0 ? step_ms : 50.0;
        return base * (1.0 + 0.45 * k);
    }
    bool want_draft() {
        if (plain_left > 0) { --plain_left; return false; }
        return true;
    }
    /// accept[i]: the confidence that draft i is accepted when the drafts before it are; n drafts, at most k_max used
    int choose(const float* accept, int n, int k_max) const {
        int best = 0;
        double e = 1, s = 1, best_rate = 1 / verify_cost(0);
        for (int k = 1; k <= std::min({k_max, n, kMax - 1}); ++k) {
            s *= accept[k - 1];
            e += s;
            if (e / verify_cost(k) > best_rate) { best_rate = e / verify_cost(k); best = k; }
        }
        return best;
    }
    void observe_step(double ms) { step_ms = step_ms > 0 ? 0.8 * step_ms + 0.2 * ms : ms; }
    void observe_round(int k, double verify_ms, double round_ms, int tokens) {
        vms[k] = vn[k] > 0 ? 0.8 * vms[k] + 0.2 * verify_ms : verify_ms;
        vn[k] += 1;
        const double rate = tokens / round_ms;
        spec_rate = spec_rate > 0 ? 0.7 * spec_rate + 0.3 * rate : rate;
        if (!always && step_ms > 0 && spec_rate < 1.0 / step_ms) {   // drafting loses: plain steps for a while
            plain_left = backoff;
            backoff = std::min(backoff * 2, 64);
        } else {
            backoff = 4;
        }
    }
};

/// A verify round's output: next[t] follows window[0..t]. The drafts window[1..] that match are kept; the round emits
/// next[0..keep) (the accepted drafts, then the correction), ended early after the first end-of-sequence token.
/// Returns keep: the inputs to commit, which is also the number of tokens emitted.
inline int round_keep(const std::vector<int>& window, const std::vector<int>& next, const std::vector<int>& eos) {
    int keep = 1;
    while (keep < (int) window.size() && window[keep] == next[keep - 1]) ++keep;
    for (int i = 0; i < keep; ++i)
        if (std::find(eos.begin(), eos.end(), next[i]) != eos.end()) return i + 1;
    return keep;
}

}  // namespace strata::ds41
