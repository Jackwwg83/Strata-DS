// include/strata/ds41/parallel.hpp - run n parts of a job on n threads (part 0 on the calling thread), safely.
//
// A thread that cannot start (std::system_error, e.g. at the process's thread limit) does not end the program: its part
// and every later part run on the calling thread. Every started thread is joined before run_parallel returns or throws;
// the first exception of a part is rethrown then.
#pragma once

#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

namespace strata::ds41 {
namespace detail {
/// tests: the start of thread number k (0 = the first one started) throws; -1 = never
inline std::atomic<int>& spawn_fault() {
    static std::atomic<int> k{-1};
    return k;
}
}  // namespace detail

template <class Fn>
void run_parallel(size_t n, Fn fn) {
    if (n == 0) return;
    std::exception_ptr error;
    std::mutex mu;
    auto part = [&](size_t i) {
        try {
            fn(i);
        } catch (...) {
            std::lock_guard<std::mutex> lk(mu);
            if (!error) error = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    struct Join {
        std::vector<std::thread>& p;
        ~Join() {
            for (auto& t : p)
                if (t.joinable()) t.join();
        }
    } join{pool};
    size_t i = 1;
    try {
        pool.reserve(n - 1);
        for (; i < n; ++i) {
            if (detail::spawn_fault().load() == (int) (i - 1))
                throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
            pool.emplace_back(part, i);
        }
    } catch (const std::system_error&) {
        for (; i < n; ++i) part(i);   // the parts no thread took
    }
    part(0);
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

}  // namespace strata::ds41
