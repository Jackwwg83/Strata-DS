#include "test_validation.hpp"
#include <cstdio>
using namespace ds41test;
int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
        failures += !ok;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    check(max_error(0, 0.001) == 0.001 && max_error(0.002, 0.001) == 0.002, "finite errors unchanged");
    for (double bad : {nan, inf, -inf}) {
        check(!(max_error(0, bad) <= 0.01), "non-finite error fails tolerance");
        check(!(max_error(max_error(0, bad), 0) <= 0.01), "later finite error cannot hide failure");
        check(!(max_error(bad, 0) <= 0.01), "non-finite accumulator fails tolerance");
    }
    const std::vector<int32_t> want{128, 129, 130, 131}, dup(4, 128), reverse{131, 130, 129, 128};
    const std::vector<int32_t> low{127, 129, 130, 131}, high{128, 129, 130, 132};
    check(valid_topk(want.data(), 4, 128, 4), "strict valid IDs accepted");
    check(topk_overlap(want.data(), 4, want) == 4, "complete set intersection");
    check(!valid_topk(dup.data(), 4, 128, 4), "duplicate IDs rejected");
    check(topk_overlap(dup.data(), 4, want) == 1, "duplicates count once in set intersection");
    check(!valid_topk(reverse.data(), 4, 128, 4), "descending IDs rejected");
    check(!valid_topk(low.data(), 4, 128, 4) && !valid_topk(high.data(), 4, 128, 4), "out-of-range IDs rejected");
    const int32_t padded[]{128, 130, -1, -1};
    check(valid_topk(padded, 2, 128, 4) && valid_topk(padded, 0, 128, 0), "valid prefix and empty row accepted");
    check(!valid_topk(padded, 4, 128, 4), "padding cannot enter valid prefix");
    std::printf("RESULT %s (%d failures)\n", failures ? "fail" : "pass", failures);
    return failures ? 1 : 0;
}
