// src/ds41/pack.cpp - read the ds41 pack written by tools/ds41/pack.py.
#include "strata/ds41/pack.hpp"

#include "strata/ds41/config.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace strata::ds41 {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("ds41 pack: " + what); }

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}

DType parse_dtype(const std::string& s) {
    if (s == "f8e4m3") return DType::F8E4M3;
    if (s == "e8m0") return DType::E8M0;
    if (s == "bf16") return DType::BF16;
    if (s == "f32") return DType::F32;
    if (s == "f16") return DType::F16;
    if (s == "i32") return DType::I32;
    if (s == "i16") return DType::I16;
    if (s == "i8") return DType::I8;
    if (s == "u8") return DType::U8;
    fail("unknown dtype " + s);
}

std::ifstream open_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) fail("cannot open " + path);
    return f;
}

bool contains(uint64_t limit, uint64_t offset, uint64_t bytes) {
    return offset <= limit && bytes <= limit - offset;
}

uint64_t component_number(const std::string& text) {
    if (text.empty() || text[0] == '-') fail("bad component size");
    size_t used = 0;
    const auto n = std::stoull(text, &used);
    if (used != text.size()) fail("bad component size");
    return n;
}

bool skip(const std::string& line) { return line.empty() || line[0] == '#'; }

const char* kComp[12] = {"w1.trellis", "w1.suh", "w1.svh", "w1.mul1", "w3.trellis", "w3.suh",
                         "w3.svh",     "w3.mul1", "w2.trellis", "w2.suh", "w2.svh", "w2.mul1"};

}  // namespace

size_t dtype_size(DType t) {
    switch (t) {
        case DType::F8E4M3: case DType::E8M0: case DType::I8: case DType::U8: return 1;
        case DType::BF16: case DType::F16: case DType::I16: return 2;
        case DType::F32: case DType::I32: return 4;
    }
    return 0;
}

Pack::Pack(const std::string& dir) : dir_(dir) {
    // pack_info.txt is written last: without "finished 1" the pack is incomplete
    {
        auto f = open_text(dir + "/pack_info.txt");
        std::string key, line;
        bool finished = false;
        while (std::getline(f, line)) {
            std::istringstream s(line);
            long long v = 0;
            s >> key >> v;
            if (key == "finished") finished = v == 1;
            if (key == "layers") n_layers_ = (int) v;
            if (key == "experts") n_experts_ = (int) v;
        }
        if (!finished) fail(dir + " is not a finished pack (pack_info.txt has no 'finished 1')");
        if (n_layers_ != kLayers || n_experts_ != kExperts) fail("layer or expert count does not match config.hpp");
    }
    {
        auto f = open_text(dir + "/index.txt");
        std::string line;
        while (std::getline(f, line)) {
            if (skip(line)) continue;
            std::istringstream s(line);
            DenseTensor t;
            std::string dt;
            int ndim = 0;
            s >> t.name >> dt >> ndim;
            if (!s || ndim < 0 || ndim > 16) fail("bad tensor rank: " + line);
            t.dtype = parse_dtype(dt);
            t.shape.resize(ndim);
            for (auto& d : t.shape) s >> d;
            s >> t.file_offset >> t.bytes;
            if (!s) fail("bad index.txt line: " + line);
            uint64_t bytes = dtype_size(t.dtype);
            for (auto d : t.shape) {
                if (d <= 0 || (uint64_t) d > std::numeric_limits<uint64_t>::max() / bytes)
                    fail("invalid or overflowing shape for " + t.name);
                bytes *= (uint64_t) d;
            }
            if (bytes != t.bytes) fail("size mismatch for " + t.name);
            dense_[t.name] = t;
        }
    }
    {
        experts_.resize((size_t) n_layers_ * n_experts_);
        std::vector<bool> seen(experts_.size(), false);
        auto f = open_text(dir + "/experts.txt");
        std::string line;
        while (std::getline(f, line)) {
            if (skip(line)) continue;
            std::istringstream s(line);
            int l = 0, e = 0;
            ExpertSlot x;
            s >> l >> e >> x.offset >> x.bytes >> x.bits[0] >> x.bits[1] >> x.bits[2];
            for (int c = 0; c < 12; ++c) {
                std::string item;
                s >> item;
                const auto a = item.find(':'), b = item.rfind(':');
                if (a == std::string::npos || b <= a || item.substr(0, a) != kComp[c])
                    fail("bad component in experts.txt: " + item);
                x.comp_off[c] = component_number(item.substr(a + 1, b - a - 1));
                x.comp_bytes[c] = component_number(item.substr(b + 1));
                if (!contains(x.bytes, x.comp_off[c], x.comp_bytes[c])) fail("component outside its slot: " + line);
            }
            if (!s || l < 0 || l >= n_layers_ || e < 0 || e >= n_experts_) fail("bad experts.txt line: " + line);
            // The tiers round slot sizes up to 4 KiB.
            if (x.bytes > std::numeric_limits<size_t>::max() - 4095)
                fail("slot size cannot be aligned");
            experts_[(size_t) l * n_experts_ + e] = x;
            seen[(size_t) l * n_experts_ + e] = true;
        }
        for (bool b : seen)
            if (!b) fail("experts.txt does not list every expert");
    }
    {
        auto f = open_text(dir + "/engram.txt");
        std::string line;
        while (std::getline(f, line)) {
            if (skip(line)) continue;
            std::istringstream s(line);
            EngramTable t;
            s >> t.layer >> t.rows >> t.dim >> t.weight_offset >> t.scale_offset;
            std::getline(s >> std::ws, t.path);
            if (!s || t.path.empty() || t.rows <= 0 || t.dim <= 0) fail("bad engram.txt line: " + line);
            engram_.push_back(t);
        }
    }
    {
        auto f = open_text(dir + "/engram_hash.txt");
        const size_t tables = engram_.size();
        hash_.multipliers.resize(tables);
        hash_.primes.resize(tables);
        hash_.offsets.resize(tables);
        std::string line;
        while (std::getline(f, line)) {
            if (skip(line)) continue;
            std::istringstream s(line);
            std::string key;
            s >> key;
            if (key == "max_ngram") s >> hash_.max_ngram;
            else if (key == "n_heads") s >> hash_.n_heads;
            else if (key == "pad") s >> hash_.pad;
            else if (key == "vocab") { long long v; s >> v; }
            else if (key == "layers") {
                int l;
                while (s >> l) hash_.layers.push_back(l);
                if (!s.eof()) fail("bad engram hash layers");
                continue;
            } else {
                int li = 0;
                if (!(s >> li) || li < 0 || (size_t) li >= tables) fail("bad engram hash table index");
                std::vector<int64_t> v;
                long long x;
                while (s >> x) v.push_back(x);
                auto& dst = key == "multipliers" ? hash_.multipliers : key == "primes" ? hash_.primes : hash_.offsets;
                if (key != "multipliers" && key != "primes" && key != "offsets") fail("bad engram_hash.txt key " + key);
                if (!s.eof() || !dst[li].empty()) fail("bad or repeated engram hash array");
                dst[li] = v;
                continue;
            }
            if (!s) fail("bad engram hash scalar: " + line);
        }
        if (hash_.layers.size() != tables) fail("engram hash layer count mismatch");
        if (tables) {
            // The engine has room for 24 row IDs per table and token.
            if (hash_.max_ngram < 2 || hash_.max_ngram > 25 || hash_.n_heads < 1 ||
                hash_.n_heads > 24 / (hash_.max_ngram - 1)) fail("bad engram hash dimensions");
            const size_t cols = (size_t) (hash_.max_ngram - 1) * hash_.n_heads;
            for (size_t i = 0; i < tables; ++i) {
                const int layer = hash_.layers[i];
                if (layer < 0 || layer >= n_layers_ || layer != engram_[i].layer ||
                    (i && layer <= hash_.layers[i - 1])) fail("engram hash layer mismatch");
                if (hash_.multipliers[i].size() != (size_t) hash_.max_ngram ||
                    hash_.primes[i].size() != cols || hash_.offsets[i].size() != cols)
                    fail("missing or incomplete engram hash array");
                for (size_t c = 0; c < cols; ++c) {
                    const int64_t prime = hash_.primes[i][c], off = hash_.offsets[i][c];
                    if (prime <= 0 || off < 0 || !contains((uint64_t) engram_[i].rows, off, prime))
                        fail("bad engram hash prime or offset");
                }
            }
        }
        std::ifstream tm(dir + "/engram_tokenmap.bin", std::ios::binary | std::ios::ate);
        if (!tm) fail("cannot open engram_tokenmap.bin");
        const auto n = (size_t) tm.tellg() / 4;
        if (n != (size_t) kVocab) fail("engram token map has the wrong size");
        hash_.token_map.resize(n);
        tm.seekg(0);
        tm.read(reinterpret_cast<char*>(hash_.token_map.data()), (std::streamsize) (n * 4));
    }
    // shape spot checks: a different model fails here, not in the logits
    auto expect = [&](const char* n, std::vector<int64_t> shape) {
        if (dense(n).shape != shape) fail(std::string("unexpected shape for ") + n);
    };
    expect("embed.weight", {kVocab, kDim});
    expect("head.weight", {kVocab, kDim});
    expect("layers.0.attn.wq_b.weight", {kHeads * kHeadDim, kQLora});
    expect("layers.0.attn.wo_a.weight", {kOGroups * kOLora, kHeads * kHeadDim / kOGroups});
    expect("layers.0.hc_attn_fn", {kHcMix, kHc * kDim});
}

Pack::~Pack() {
    if (arena_) cudaFree(arena_);
    if (experts_map_) munmap(const_cast<uint8_t*>(experts_map_), experts_map_bytes_);
}

const DenseTensor& Pack::dense(const std::string& name) const {
    auto it = dense_.find(name);
    if (it == dense_.end()) fail("no dense tensor " + name);
    return it->second;
}

uint64_t Pack::upload_dense() {
    const std::string path = dir_ + "/dense.bin";
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + path);
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        fail("cannot size dense.bin");
    }
    const uint64_t total = (uint64_t) st.st_size;
    for (const auto& kv : dense_) {
        if (!contains(total, kv.second.file_offset, kv.second.bytes)) {
            close(fd);
            fail("tensor past end of dense.bin: " + kv.first);
        }
    }
    check_cuda(cudaMalloc(&arena_, total), "cudaMalloc dense arena");
    constexpr uint64_t kChunk = 64ull << 20;
    void* host = nullptr;
    check_cuda(cudaMallocHost(&host, kChunk), "cudaMallocHost staging");
    for (uint64_t off = 0; off < total; off += kChunk) {
        const uint64_t n = std::min(kChunk, total - off);
        uint64_t got = 0;
        while (got < n) {
            const ssize_t r = pread(fd, (char*) host + got, n - got, (off_t) (off + got));
            if (r <= 0) fail("short read of dense.bin");
            got += (uint64_t) r;
        }
        check_cuda(cudaMemcpy((char*) arena_ + off, host, n, cudaMemcpyHostToDevice), "upload dense");
    }
    cudaFreeHost(host);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // its 11 GB stay out of the file cache the experts need
    close(fd);
    for (auto& kv : dense_) {
        kv.second.device = (char*) arena_ + kv.second.file_offset;
    }
    return total;
}

void Pack::map_experts() {
    const std::string path = dir_ + "/experts.bin";
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + path);
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        fail("cannot size experts.bin");
    }
    const uint64_t total = (uint64_t) st.st_size;
    for (const auto& expert : experts_) {
        if (!contains(total, expert.offset, expert.bytes)) {
            close(fd);
            fail("expert past end of experts.bin");
        }
    }
    void* p = mmap(nullptr, total, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) fail("mmap experts.bin failed");
    if (experts_map_) munmap(const_cast<uint8_t*>(experts_map_), experts_map_bytes_);
    experts_map_ = (const uint8_t*) p;
    experts_map_bytes_ = total;
}

}  // namespace strata::ds41
