// Standalone CPU-only checks of the production K14 workspace and validation API.
// Link with k14_indexer_prefill.o and cudart; this test never launches a kernel.
#include "strata/ds41/kernels/k14_indexer_prefill.hpp"
#include <cassert>
#include <climits>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace kk = strata::ds41::kernels;
int main() {
    int checks = 0;
    auto equal = [&](size_t got, size_t want) { assert(got == want); ++checks; };
    auto invalid = [&](auto fn) {
        bool caught = false;
        try { fn(); } catch (const std::invalid_argument&) { caught = true; }
        assert(caught);
        ++checks;
    };
    for (int m : {1, 3, 4, 255, 256, 257, 4096, 16384}) {
        for (int64_t t : {int64_t(0), int64_t(1), int64_t(63), int64_t(64), int64_t(65), int64_t(8192),
                          int64_t(INT_MAX) + 16384}) {
            equal(kk::indexer_topk_prefill_workspace_bytes(m, t),
                  t ? size_t(m < 256 ? m : 256) * size_t(t) * 2 + 255 : 0);
        }
    }
    invalid([&] { kk::indexer_topk_prefill_workspace_bytes(0, 1); });
    invalid([&] { kk::indexer_topk_prefill_workspace_bytes(16385, 1); });
    invalid([&] { kk::indexer_topk_prefill_workspace_bytes(1, -1); });
    invalid([&] { kk::indexer_topk_prefill_workspace_bytes(1, INT64_MAX); });
    invalid([&] { kk::indexer_topk_prefill_workspace_bytes(256, INT64_MAX / 256); });
    invalid([&] { kk::indexer_topk_prefill_workspace_bytes(1, INT64_MAX / 2); });

    // Non-null sentinels are only passed to calls rejected before device access.
    auto* bf = reinterpret_cast<const __nv_bfloat16*>(uintptr_t(256));
    auto* bytes = reinterpret_cast<uint8_t*>(uintptr_t(256));
    auto* out = reinterpret_cast<int32_t*>(uintptr_t(256));
    auto call = [&](int m, int pos, int ratio, int k, int offset, int block,
                    const uint8_t* ci, uint8_t* co, int64_t stride, void* ws, size_t size) {
        kk::indexer_topk_prefill(bf, bf, bf, m, pos, ratio, ci, co, stride, k, offset,
                                64, block, out, ws, size, nullptr);
    };
    const size_t enough = kk::indexer_topk_prefill_workspace_bytes(2, 1002);
    invalid([&] { call(0, 0, 1, 512, 128, 8, nullptr, nullptr, 0, bytes, enough); });
    invalid([&] { call(16385, 0, 1, 512, 128, 8, nullptr, nullptr, 0, bytes, enough); });
    invalid([&] { call(2, -1, 1, 512, 128, 8, nullptr, nullptr, 0, bytes, enough); });
    invalid([&] { call(2, 1000, 0, 512, 128, 8, nullptr, nullptr, 0, bytes, enough); });
    invalid([&] { call(2, 1000, 1, 0, 128, 8, nullptr, nullptr, 0, bytes, enough); });
    invalid([&] { call(2, 1000, 1, 512, 128, 0, nullptr, bytes, 1002, bytes, enough); });
    invalid([&] { call(2, 1000, 1, 512, 128, 8, bytes, nullptr, 1001, bytes, enough); });
    invalid([&] { call(2, 1000, 1, 512, 128, 8, nullptr, bytes, -1, bytes, enough); });
    invalid([&] { call(2, 1000, 1, 512, 128, 8, nullptr, bytes, INT64_MAX, bytes, enough); });
    invalid([&] { call(2, 1000, 1, 512, 128, 8, nullptr, nullptr, 0, bytes, enough - 1); });
    invalid([&] { call(2, 1000, 1, 512, 128, 8, nullptr, nullptr, 0, nullptr, enough); });
    invalid([&] { call(2, 1000, 1, 512, INT_MAX, 8, nullptr, nullptr, 0, bytes, enough); });
    invalid([&] {
        kk::indexer_topk_prefill(nullptr, bf, bf, 2, 1000, 1, nullptr, nullptr, 0,
                                512, 128, 0, 8, out, bytes, enough, nullptr);
    });
    invalid([&] {
        kk::indexer_topk_prefill(bf, bf, bf, 2, 1000, 1, nullptr, nullptr, 0,
                                512, 128, 0, 8, nullptr, bytes, enough, nullptr);
    });
    const size_t near_limit = kk::indexer_topk_prefill_workspace_bytes(16384, int64_t(INT_MAX) + 16384);
    invalid([&] { call(16384, INT_MAX, 1, 512, 0, 8, nullptr, nullptr, 0, bytes, near_limit); });
    std::printf("PASS: %d production host contract checks; no device calls\n", checks);
}
