// src/ds41/tests/parallel_test.cpp - run_parallel: every part runs exactly once, also when a thread cannot start
// (the part then runs on the calling thread), and an exception in a part reaches the caller after every thread ended.
// ThreadPool: the same contract with threads that persist between runs.
//   c++ -std=c++17 -Iinclude src/ds41/tests/parallel_test.cpp -o /tmp/pt -lpthread && /tmp/pt
#include "strata/ds41/parallel.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <thread>
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

    // ThreadPool: helper threads started once; a helper that cannot start leaves its parts to the caller
    for (int fail_at : {-1, 0, 3}) {
        detail::spawn_fault() = fail_at;
        ThreadPool pool(8);
        detail::spawn_fault() = -1;
        check(pool.size() == (fail_at < 0 ? 8u : 1u + (size_t) fail_at),
              "pool size counts the caller and the helpers that started (start " + std::to_string(fail_at) + " fails)");
        bool once = true;
        for (int round = 0; round < 2000 && once; ++round) {
            const size_t n = 1 + round % 12;   // fewer, as many and more parts than threads
            std::vector<std::atomic<int>> runs(n);
            pool.run(n, [&](size_t i) { ++runs[i]; });
            for (auto& r : runs) once &= r.load() == 1;
        }
        check(once, "2000 pool runs of 1..12 parts: every part ran exactly once (start " + std::to_string(fail_at) +
                        " fails)");
    }
    {
        ThreadPool pool(4);
        std::set<std::thread::id> first, second;
        std::mutex mu;
        auto record = [&](std::set<std::thread::id>& ids) {
            return [&](size_t) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));   // every part on its own thread
                std::lock_guard<std::mutex> lk(mu);
                ids.insert(std::this_thread::get_id());
            };
        };
        pool.run(4, record(first));
        pool.run(4, record(second));
        check(first.size() == 4 && first == second, "a pool runs its parts on the same threads every time");
        check(first.count(std::this_thread::get_id()) == 1, "part 0 runs on the caller");
        std::atomic<int> late{0};
        bool caught = false;
        try {
            pool.run(6, [&](size_t i) {
                if (i == 2) throw std::runtime_error("part 2");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                ++late;
            });
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()) == "part 2";
        }
        check(caught && late.load() == 5, "a pool part's exception reaches the caller after the other parts ran");
        std::atomic<int> again{0};
        pool.run(4, [&](size_t) { ++again; });
        check(again.load() == 4, "the pool runs again after an exception");
        pool.run(0, [&](size_t) { check(false, "no pool part for n = 0"); });
    }
    {   // a helper start that fails with something else than system_error: the started helpers are joined, the
        // exception reaches the caller (no std::terminate from a joinable thread)
        detail::spawn_alloc_fault() = 3;
        bool caught = false;
        try {
            ThreadPool pool(8);
        } catch (const std::bad_alloc&) {
            caught = true;
        }
        detail::spawn_alloc_fault() = -1;
        check(caught, "a ThreadPool whose fourth helper cannot be allocated throws bad_alloc after joining three");
    }
    std::printf("RESULT %s parallel\n", failures ? "fail" : "pass");
    return failures ? 1 : 0;
}
