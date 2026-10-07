// Host-only acceptance helpers. Use these before aggregating kernel errors.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

namespace ds41test {
inline double max_error(double worst, double error) {
    if (!std::isfinite(worst) || !std::isfinite(error))
        return std::numeric_limits<double>::infinity();
    return std::max(worst, error);
}
inline bool valid_topk(const int32_t* row, int count, int64_t offset, int64_t length) {
    for (int j = 0; j < count; ++j) {
        const int64_t id = row[j];
        if (id < offset || id - offset >= length || (j > 0 && row[j - 1] >= id)) return false;
    }
    return true;
}
inline int64_t topk_overlap(const int32_t* row, int count, const std::vector<int32_t>& want) {
    const std::set<int32_t> expected(want.begin(), want.end());
    const std::set<int32_t> actual(row, row + count);
    int64_t overlap = 0;
    for (int32_t id : actual) overlap += expected.count(id);
    return overlap;
}
}  // namespace ds41test
