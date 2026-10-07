// src/ds41/tests/skip_misses_test.cpp - skip_misses_row (DS41_SKIP_MISS): which routed experts a decode step leaves
// out, and the weights of the rest. Host only: the kernel runs the same function.
//   c++ -std=c++17 -Iinclude src/ds41/tests/skip_misses_test.cpp -o /tmp/smt && /tmp/smt
#include "strata/ds41/skip_misses.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace strata::ds41;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
bool near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

}  // namespace

int main() {
    // 8 experts; 2, 5 and 7 are in VRAM
    const std::vector<int32_t> res = {-1, -1, 0, -1, -1, 1, -1, 2};
    {
        // weights 0.5 0.4 0.3 0.2 0.05 0.05 (sum 1.5): with tau 0.1, misses below 0.15 are left out
        const int32_t ids[6] = {0, 2, 1, 3, 4, 5};
        float w[6] = {0.5f, 0.4f, 0.3f, 0.2f, 0.05f, 0.05f};
        int32_t out[6];
        const int n = skip_misses_row(ids, w, 6, res.data(), 0.1f, false, out);
        check(n == 1, "one expert left out");
        check(out[0] == 0 && out[1] == 2 && out[2] == 1 && out[3] == 3 && out[4] == -1 && out[5] == 5,
              "a low-weight miss is left out; a low-weight VRAM hit (5) and heavier misses stay");
        check(near(w[0], 0.5f) && near(w[4], 0.05f), "without renormalization the weights do not change");
    }
    {
        const int32_t ids[6] = {0, 2, 1, 3, 4, 6};
        float w[6] = {0.5f, 0.4f, 0.3f, 0.2f, 0.05f, 0.05f};
        int32_t out[6];
        const int n = skip_misses_row(ids, w, 6, res.data(), 0.1f, true, out);
        check(n == 2 && out[4] == -1 && out[5] == -1, "two low-weight misses left out");
        const float scale = 1.5f / 1.4f;
        check(near(w[0], 0.5f * scale) && near(w[3], 0.2f * scale), "the kept weights scale to the old sum");
        float kept = 0;
        for (int j = 0; j < 6; ++j)
            if (out[j] >= 0) kept += w[j];
        check(near(kept, 1.5f), "the kept weights sum to the old sum");
    }
    {
        // every expert a miss with the same weight: the heaviest (first of the ties) always stays
        const int32_t ids[6] = {0, 1, 3, 4, 6, 6};
        float w[6] = {0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f};
        int32_t out[6];
        const int n = skip_misses_row(ids, w, 6, res.data(), 0.9f, true, out);
        check(n == 5 && out[0] == 0, "with a tau above every weight, only the heaviest expert stays");
        check(near(w[0], 1.5f), "and it carries the whole weight");
    }
    {
        const int32_t ids[6] = {0, 1, 3, 4, 6, 6};
        float w[6] = {0.5f, 0.4f, 0.3f, 0.2f, 0.05f, 0.05f};
        int32_t out[6];
        check(skip_misses_row(ids, w, 6, res.data(), 0.0f, true, out) == 0, "tau 0 leaves nothing out");
        bool same = true;
        for (int j = 0; j < 6; ++j) same &= out[j] == ids[j];
        check(same && near(w[0], 0.5f), "and keeps ids and weights");
    }
    {
        // no VRAM tier: every expert is a miss; a negative id stays negative and is not counted
        const int32_t ids[6] = {0, 2, -1, 3, 4, 5};
        float w[6] = {0.5f, 0.4f, 0.3f, 0.2f, 0.05f, 0.05f};
        int32_t out[6];
        const int n = skip_misses_row(ids, w, 6, nullptr, 0.1f, false, out);
        check(n == 2 && out[2] == -1 && out[4] == -1 && out[5] == -1 && out[1] == 2,
              "without a VRAM table every expert counts as a miss");
    }
    std::printf("RESULT %s skip_misses\n", failures ? "fail" : "pass");
    return failures ? 1 : 0;
}
