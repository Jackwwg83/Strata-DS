// include/strata/ds41/pack.hpp - the ds41 pack as the engine sees it (written by tools/ds41/pack.py).
//
// The loader reads only flat text indexes; it never re-derives a layout (same rule as upstream's weights.hpp).
//   dense:   every non-expert tensor copied into one device arena, looked up by name
//   experts: experts.bin mapped read-only; each expert is a 4 KiB aligned slot with 12 components
//   engram:  the two n-gram tables stay in their original files; rows are read on demand
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::ds41 {

enum class DType { F8E4M3, E8M0, BF16, F32, F16, I32, I16, I8, U8 };

struct DenseTensor {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    uint64_t file_offset = 0;
    uint64_t bytes = 0;
    void* device = nullptr;          ///< inside the arena, after upload
};

/// One routed expert: offsets are inside its slot.
struct ExpertSlot {
    uint64_t offset = 0;             ///< in experts.bin
    uint64_t bytes = 0;
    float bits[3] = {0, 0, 0};       ///< w1, w3, w2
    uint64_t comp_off[12] = {};      ///< w1.trellis w1.suh w1.svh w1.mul1, then w3, then w2
    uint64_t comp_bytes[12] = {};
};

struct EngramTable {
    int layer = 0;
    int64_t rows = 0;
    int dim = 0;
    uint64_t weight_offset = 0;
    uint64_t scale_offset = 0;
    std::string path;
};

struct EngramHash {
    int max_ngram = 0;
    int n_heads = 0;
    int64_t pad = 0;
    std::vector<int> layers;                        ///< backbone layer of each engram table
    std::vector<std::vector<int64_t>> multipliers;  ///< [table][max_ngram]
    std::vector<std::vector<int64_t>> primes;       ///< [table][(max_ngram-1)*n_heads]
    std::vector<std::vector<int64_t>> offsets;      ///< [table][(max_ngram-1)*n_heads]
    std::vector<int32_t> token_map;                 ///< vocab -> compressed id
};

class Pack {
public:
    /// Parses the indexes and checks that the pack is finished. Throws on any inconsistency.
    explicit Pack(const std::string& dir);
    ~Pack();
    Pack(const Pack&) = delete;
    Pack& operator=(const Pack&) = delete;

    /// Copies dense.bin into one device arena. Returns the arena size in bytes.
    uint64_t upload_dense();
    /// Maps experts.bin read-only (MAP_SHARED: pages come from the OS page cache, as upstream's mmap tier).
    void map_experts();

    const DenseTensor& dense(const std::string& name) const;
    bool has_dense(const std::string& name) const { return dense_.count(name) != 0; }
    const ExpertSlot& expert(int layer, int e) const { return experts_[(size_t) layer * n_experts_ + e]; }
    const uint8_t* expert_base() const { return experts_map_; }
    const std::vector<EngramTable>& engram_tables() const { return engram_; }
    const EngramHash& engram_hash() const { return hash_; }
    int n_layers() const { return n_layers_; }
    int n_experts() const { return n_experts_; }
    const std::string& dir() const { return dir_; }

private:
    std::string dir_;
    int n_layers_ = 0;
    int n_experts_ = 0;
    std::map<std::string, DenseTensor> dense_;
    std::vector<ExpertSlot> experts_;
    std::vector<EngramTable> engram_;
    EngramHash hash_;
    void* arena_ = nullptr;
    const uint8_t* experts_map_ = nullptr;
    uint64_t experts_map_bytes_ = 0;
};

size_t dtype_size(DType t);

}  // namespace strata::ds41
