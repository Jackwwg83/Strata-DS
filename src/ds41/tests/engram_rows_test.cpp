// src/ds41/tests/engram_rows_test.cpp - EngramRows returns exactly the bytes a plain pread returns, on the O_DIRECT
// path (a file in the working directory, where the file system allows it) and on the fallback path (/dev/shm is
// tmpfs, which refuses O_DIRECT). Offsets are unaligned, so some rows straddle a 4 KiB boundary.
#include "strata/ds41/engram_rows.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using strata::ds41::EngramRows;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what.c_str()); }
}

/// Two tables in one file each; returns false when the directory cannot hold the file.
bool run(const std::string& dir, const char* name) {
    const int rows = 4000, R = 24;
    const uint64_t w_off = 1234, s_off = w_off + (uint64_t) rows * 256 + 777;
    const size_t size = s_off + (size_t) rows * 8 + 99;
    std::vector<std::string> paths;
    std::vector<std::vector<uint8_t>> data;
    for (int t = 0; t < 2; ++t) {
        std::vector<uint8_t> d(size);
        std::mt19937 g(17 + t);
        for (auto& b : d) b = (uint8_t) g();
        const std::string p = dir + "/ds41_engram_rows_test_" + std::to_string(t) + ".bin";
        std::FILE* f = std::fopen(p.c_str(), "wb");
        if (!f) return false;
        std::fwrite(d.data(), 1, d.size(), f);
        std::fclose(f);
        paths.push_back(p);
        data.push_back(std::move(d));
    }
    for (int threads : {0, 64}) {   // the default queue depth and prefill's
        EngramRows er({{paths[0], w_off, s_off}, {paths[1], w_off, s_off}}, R, 256, 8, threads);
        std::printf("%s, %d threads: %s\n", name, threads, er.direct() ? "O_DIRECT" : "plain pread fallback");
        std::mt19937 g(5);
        for (int round = 0; round < 20; ++round) {
            std::vector<std::vector<int64_t>> ids(2, std::vector<int64_t>(R));
            for (auto& v : ids)
                for (auto& x : v) x = (int64_t) (g() % rows);
            ids[0][0] = 15;                  // 1234 + 15*256 = 5074: inside one block
            ids[0][1] = (4096 - 1234) / 256; // 11: 1234 + 2816 = 4050, the row crosses 4096
            ids[1][R - 1] = rows - 1;        // the last row
            std::vector<uint8_t> w(2 * R * 256), s(2 * R * 8);
            er.read({ids[0].data(), ids[1].data()}, R, {w.data(), w.data() + R * 256}, {s.data(), s.data() + R * 8});
            for (int t = 0; t < 2; ++t)
                for (int i = 0; i < R; ++i) {
                    const uint8_t* want_w = data[t].data() + w_off + ids[t][i] * 256;
                    const uint8_t* want_s = data[t].data() + s_off + ids[t][i] * 8;
                    check(std::memcmp(w.data() + (t * R + i) * 256, want_w, 256) == 0,
                          std::string(name) + " weights row " + std::to_string(ids[t][i]));
                    check(std::memcmp(s.data() + (t * R + i) * 8, want_s, 8) == 0,
                          std::string(name) + " scales row " + std::to_string(ids[t][i]));
                }
        }
    }
    for (const auto& p : paths) std::remove(p.c_str());
    return true;
}

}  // namespace

int main() {
    check(run(".", "working directory"), "working directory is writable");
    if (!run("/dev/shm", "/dev/shm")) std::printf("/dev/shm not available: fallback path not tested\n");
    std::printf("RESULT %s\n", failures ? "fail" : "pass");
    return failures ? 1 : 0;
}
