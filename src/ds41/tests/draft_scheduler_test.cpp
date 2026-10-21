// src/ds41/tests/draft_scheduler_test.cpp - DSpark's policy (include/strata/ds41/draft_scheduler.hpp) on the host:
// the cold start, the choice of k, the backoff, and what a verify round emits.
#include "strata/ds41/draft_scheduler.hpp"

#include <cstdio>
#include <string>

using strata::ds41::DraftScheduler;
using strata::ds41::round_keep;

namespace {
int fails = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++fails;
        std::printf("FAIL: %s\n", what.c_str());
    }
}
}  // namespace

int main() {
    {   // the first 16 tokens are plain steps, then drafting
        DraftScheduler s;
        int plain = 0;
        while (!s.want_draft()) ++plain;
        check(plain == 16, "16 plain steps first");
        check(s.want_draft(), "then drafts");
    }
    {   // k: unmeasured costs grow 45% a row; confident drafts are worth it, unsure ones are not
        DraftScheduler s;
        s.observe_step(100);
        const float sure[5] = {0.95f, 0.95f, 0.95f, 0.95f, 0.95f};
        const float unsure[5] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
        check(s.choose(sure, 5, 3) == 3, "sure drafts: all three allowed");
        check(s.choose(sure, 2, 3) == 2, "no more than the drafts there are");
        check(s.choose(unsure, 5, 3) == 0, "unsure drafts: a plain row");
        // a measured k replaces the guess: a cheap 2-draft window wins over 3
        s.observe_round(2, 110, 120, 3);
        s.observe_round(3, 400, 410, 4);
        check(s.choose(sure, 5, 3) == 2, "measured verify times decide");
    }
    {   // a losing round backs off 4, then 8; a winning round resets the backoff
        DraftScheduler s;
        for (int i = 0; i < 16; ++i) s.want_draft();
        s.observe_step(50);
        s.observe_round(1, 200, 200, 1);   // 0.005 tok/ms against 0.02: loses
        int plain = 0;
        while (!s.want_draft()) ++plain;
        check(plain == 4, "first backoff: 4 plain steps");
        s.observe_round(1, 200, 200, 1);
        plain = 0;
        while (!s.want_draft()) ++plain;
        check(plain == 8, "second backoff: 8");
        DraftScheduler w;
        for (int i = 0; i < 16; ++i) w.want_draft();
        w.observe_step(50);
        w.observe_round(3, 60, 70, 4);     // 0.057 tok/ms: wins
        check(w.want_draft(), "a winning round keeps drafting");
    }
    {   // always: the same losing rounds leave no plain steps
        DraftScheduler s;
        s.always = true;
        for (int i = 0; i < 16; ++i) s.want_draft();
        s.observe_step(50);
        s.observe_round(1, 200, 200, 1);
        check(s.want_draft(), "always: drafting goes on after a losing round");
    }
    {   // round_keep: the matching drafts, the correction, and an early end of sequence
        const std::vector<int> eos = {1};
        check(round_keep({7, 8, 9, 10}, {8, 9, 10, 11}, eos) == 4, "all drafts accepted: 4 tokens");
        check(round_keep({7, 8, 9, 10}, {8, 5, 10, 11}, eos) == 2, "second draft wrong: 2 tokens");
        check(round_keep({7, 8, 9, 10}, {6, 9, 10, 11}, eos) == 1, "first draft wrong: the correction only");
        check(round_keep({7, 8, 1, 10}, {8, 1, 10, 11}, eos) == 2, "the end of sequence ends the round");
        check(round_keep({7}, {1}, eos) == 1, "a one-row window");
    }
    {   // round_picks: a sampled request's round picks row t only while the drafts before it matched; with the same
        // picks it keeps what round_keep keeps
        const std::vector<int> eos = {1};
        struct Case { std::vector<int> window, picks; int keep, calls; const char* what; };
        const Case cases[] = {{{7, 8, 9, 10}, {8, 9, 10, 11}, 4, 4, "all drafts match: every row picked"},
                              {{7, 8, 9, 10}, {8, 5, 10, 11}, 2, 2, "second draft differs: two rows picked"},
                              {{7, 8, 9, 10}, {6, 9, 10, 11}, 1, 1, "first draft differs: one row picked"},
                              {{7, 8, 1, 10}, {8, 1, 10, 11}, 2, 2, "the end of sequence ends the round"},
                              {{7}, {1}, 1, 1, "a one-row window"}};
        for (const Case& c : cases) {
            int calls = 0;
            std::vector<int> next;
            const int keep = strata::ds41::round_picks(c.window, [&](int t) { ++calls; return c.picks[(size_t) t]; },
                                                       eos, next);
            check(keep == c.keep && calls == c.calls, std::string("round_picks: ") + c.what);
            check(keep == round_keep(c.window, c.picks, eos), std::string("round_picks = round_keep: ") + c.what);
            check(next == std::vector<int>(c.picks.begin(), c.picks.begin() + keep), std::string("next: ") + c.what);
        }
    }
    {   // row_history: row t of a window counts penalties over the session's last h tokens, then window[0..t]
        const std::vector<int> live = {1, 2, 3, 4, 5}, window = {6, 7, 8};
        check(strata::ds41::row_history(live, window, 0, 3) == std::vector<int>({3, 4, 5, 6}), "row 0: the fed token");
        check(strata::ds41::row_history(live, window, 2, 2) == std::vector<int>({4, 5, 6, 7, 8}), "row 2: two drafts");
        check(strata::ds41::row_history(live, window, 1, 9) == std::vector<int>({1, 2, 3, 4, 5, 6, 7}), "short session");
        check(strata::ds41::row_history(live, window, 1, 0) == std::vector<int>({6, 7}), "no penalty window");
    }
    std::printf("RESULT %s\n", fails ? "fail" : "pass");
    return fails ? 1 : 0;
}
