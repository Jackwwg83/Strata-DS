// src/ds41/tests/parallel_test.cpp - run_parallel: every part runs exactly once, also when a thread cannot start
// (the part then runs on the calling thread), and an exception in a part reaches the caller after every thread ended.
//   c++ -std=c++17 -Iinclude src/ds41/tests/parallel_test.cpp -o /tmp/pt -lpthread && /tmp/pt
#include "strata/ds41/parallel.hpp"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::ds41;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
}  // namespace

int main() {
    for (int fail_at : {-1, 0, 1, 5}) {
        detail::spawn_fault() = fail_at;   // the fail_at-th thread start throws (-1: none)
        std::vector<std::atomic<int>> runs(16);
        run_parallel(runs.size(), [&](size_t i) { ++runs[i]; });
        bool once = true;
        for (auto& r : runs) once &= r.load() == 1;
        check(once, "every part ran exactly once (thread start " + std::to_string(fail_at) + " fails)");
    }
    detail::spawn_fault() = 2;
    std::atomic<int> done{0};
    bool caught = false;
    try {
        run_parallel(8, [&](size_t i) {
            if (i == 3) throw std::runtime_error("part 3");
            ++done;
        });
    } catch (const std::runtime_error& e) {
        caught = std::string(e.what()) == "part 3";
    }
    check(caught && done.load() == 7, "a part's exception reaches the caller after the other parts ran");
    detail::spawn_fault() = -1;
    run_parallel(0, [&](size_t) { check(false, "no part for n = 0"); });
    std::printf("RESULT %s parallel\n", failures ? "fail" : "pass");
    return failures ? 1 : 0;
}
