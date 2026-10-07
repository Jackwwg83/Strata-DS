#pragma once

// Shared synthetic bytes. References come from make_mixedk_golden.py.
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mixedk {
constexpr int H = 5120, F = 2304, E = 6;

inline std::string directory() {
    const char* path = std::getenv("MIXEDK_GOLDEN");
    return path ? path : "/workspace/ci/golden/mixedk";
}

template<class T>
std::vector<T> read(const std::string& path, size_t count) {
    std::vector<T> data(count);
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T)))
        throw std::runtime_error("Missing mixed-K fixture: " + path + "; run ds41/ci/make_mixedk_golden.py");
    return data;
}

struct Projection {
    int k, n, bits;
    std::vector<uint16_t> trellis, suh, svh;
    Projection(const std::string& prefix, int k_, int n_, int bits_)
        : k(k_), n(n_), bits(bits_),
          trellis(read<uint16_t>(prefix + ".trellis", size_t(k / 16) * (n / 16) * 16 * bits)),
          suh(read<uint16_t>(prefix + ".suh", k)), svh(read<uint16_t>(prefix + ".svh", n)) {}
};

inline std::vector<Projection> load() {
    const std::array<std::array<int, 3>, E> expected{{{1, 2, 6}, {2, 3, 1}, {3, 4, 2},
                                                   {4, 5, 3}, {5, 6, 4}, {6, 1, 5}}};
    const char* names[] = {"w1", "w3", "w2"};
    std::ifstream rates(directory() + "/rates.txt");
    std::vector<Projection> data;
    data.reserve(E * 3);
    for (int e = 0; e < E; ++e) for (int p = 0; p < 3; ++p) {
        int bits = 0;
        if (!(rates >> bits) || bits != expected[e][p])
            throw std::runtime_error("Invalid mixed-K rates.txt; regenerate the fixture");
        data.emplace_back(directory() + "/" + std::to_string(e) + "_" + names[p],
                          p == 2 ? F : H, p == 2 ? H : F, bits);
    }
    return data;
}
}  // namespace mixedk
