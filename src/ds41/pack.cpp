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
            t.dtype = parse_dtype(dt);
            t.shape.resize(ndim);
            for (auto& d : t.shape) s >> d;
            s >> t.file_offset >> t.bytes;
            if (!s) fail("bad index.txt line: " + line);
            int64_t n = 1;
            for (auto d : t.shape) n *= d;
            if ((uint64_t) n * dtype_size(t.dtype) != t.bytes) fail("size mismatch for " + t.name);
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
                if (a == std::string::npos || item.substr(0, a) != kComp[c]) fail("bad component in experts.txt: " + item);
                x.comp_off[c] = std::stoull(item.substr(a + 1, b - a - 1));
                x.comp_bytes[c] = std::stoull(item.substr(b + 1));
                if (x.comp_off[c] + x.comp_bytes[c] > x.bytes) fail("component outside its slot: " + line);
            }
            if (!s || l < 0 || l >= n_layers_ || e < 0 || e >= n_experts_) fail("bad experts.txt line: " + line);
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
            s >> t.layer >> t.rows >> t.dim >> t.weight_offset >> t.scale_offset >> t.path;
            if (!s) fail("bad engram.txt line: " + line);
            engram_.push_back(t);
        }
    }
    {
        auto f = open_text(dir + "/engram_hash.txt");
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
            else if (key == "layers") { int l; while (s >> l) hash_.layers.push_back(l); }
            else {
                int li = 0;
                s >> li;
                std::vector<int64_t> v;
                long long x;
                while (s >> x) v.push_back(x);
                auto& dst = key == "multipliers" ? hash_.multipliers : key == "primes" ? hash_.primes : hash_.offsets;
                if (key != "multipliers" && key != "primes" && key != "offsets") fail("bad engram_hash.txt key " + key);
                if ((int) dst.size() <= li) dst.resize(li + 1);
                dst[li] = v;
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
    fstat(fd, &st);
    const uint64_t total = (uint64_t) st.st_size;
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
    close(fd);
    for (auto& kv : dense_) {
        if (kv.second.file_offset + kv.second.bytes > total) fail("tensor past end of dense.bin: " + kv.first);
        kv.second.device = (char*) arena_ + kv.second.file_offset;
    }
    return total;
}

void Pack::map_experts() {
    const std::string path = dir_ + "/experts.bin";
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + path);
    struct stat st {};
    fstat(fd, &st);
    experts_map_bytes_ = (uint64_t) st.st_size;
    void* p = mmap(nullptr, experts_map_bytes_, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) fail("mmap experts.bin failed");
    experts_map_ = (const uint8_t*) p;
    const auto& last = experts_.back();
    if (last.offset + last.bytes > experts_map_bytes_) fail("experts.bin is shorter than experts.txt says");
}

}  // namespace strata::ds41
