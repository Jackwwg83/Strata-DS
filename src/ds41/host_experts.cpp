// src/ds41/host_experts.cpp - see include/strata/ds41/host_experts.hpp.
#include "strata/ds41/host_experts.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/residency.hpp"
#include "strata/ds41/vram_experts.hpp"

#include "moe_mul1.h"   // third_party/exllamav3_moe

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace strata::ds41 {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ds41 RAM tier: ") + what + ": " + cudaGetErrorString(e));
}

/// a "key: N kB" line of /proc/meminfo, or a "key N" line of a cgroup stat file, in bytes; 0 when absent
uint64_t read_field(const char* path, const std::string& key, uint64_t scale) {
    std::ifstream f(path);
    std::string k;
    uint64_t v = 0;
    while (f >> k >> v) {
        if (k == key || k == key + ":") return v * scale;
        std::string rest;
        std::getline(f, rest);
    }
    return 0;
}

uint64_t read_number(const char* path) {
    std::ifstream f(path);
    std::string s;
    if (!(f >> s) || s == "max") return 0;
    return std::strtoull(s.c_str(), nullptr, 10);
}

}  // namespace

std::vector<std::pair<int, int>> plan_ram_tier(const std::vector<std::pair<int, int>>& ranked,
                                               const std::vector<int32_t>& vram_res, int n_experts,
                                               const std::vector<uint64_t>& bytes, uint64_t budget) {
    std::vector<std::pair<int, int>> out;
    uint64_t at = 0;
    for (const auto& [l, e] : ranked) {
        const size_t i = (size_t) l * n_experts + e;
        if (vram_res[i] >= 0 || bytes[i] > budget - at) continue;   // upstream: skip it, smaller ones may fit
        out.emplace_back(l, e);
        at += bytes[i];
    }
    return out;
}

size_t auto_ram_budget(size_t headroom) {
    uint64_t avail = read_field("/proc/meminfo", "MemAvailable", 1024);
    uint64_t limit = read_number("/sys/fs/cgroup/memory.max"), anon = 0;
    if (limit) {
        anon = read_field("/sys/fs/cgroup/memory.stat", "anon", 1);
    } else {   // cgroup v1 (MemAvailable is then the host's): its limit, "no limit" being a huge number
        limit = read_number("/sys/fs/cgroup/memory/memory.limit_in_bytes");
        if (limit >= (1ull << 60)) limit = 0;
        anon = read_field("/sys/fs/cgroup/memory/memory.stat", "total_rss", 1);
    }
    if (limit) {   // a container: its limit counts the file cache too, which the arena may displace
        const uint64_t room = limit > anon ? limit - anon : 0;
        avail = avail ? std::min(avail, room) : room;
    }
    return avail > headroom ? (size_t) (avail - headroom) : 0;
}

HostExperts::HostExperts(const Pack& pack, const std::vector<std::pair<int, int>>& ranked,
                         const std::vector<int32_t>& vram_res, size_t budget, const std::vector<int64_t>& cpu_handles,
                         int threads)
    : pack_(pack), handles_(cpu_handles), n_experts_(pack.n_experts()) {
    const int L = pack.n_layers();
    slot_.assign((size_t) L * n_experts_, -1);
    off_.assign(1, 0);
    // each slot as large as its expert, 4 KiB aligned (a pack's experts are already)
    std::vector<uint64_t> bytes((size_t) L * n_experts_);
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < n_experts_; ++e) bytes[(size_t) l * n_experts_ + e] = (pack.expert(l, e).bytes + 4095) / 4096 * 4096;
    const auto plan = plan_ram_tier(ranked, vram_res, n_experts_, bytes, budget);
    slots_ = (int) plan.size();
    if (slots_ == 0) return;
    for (const auto& [l, e] : plan) {
        const uint64_t b = bytes[(size_t) l * n_experts_ + e];
        off_.push_back(off_.back() + b);
        max_slot_bytes_ = std::max<size_t>(max_slot_bytes_, b);
    }
    arena_bytes_ = off_.back();
    void* p = mmap(nullptr, arena_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("ds41 RAM tier: cannot reserve " + std::to_string(arena_bytes_) + " B");
    arena_ = (uint8_t*) p;
    holder_ = plan;
    // Read the slots with O_DIRECT, 16 readers: the file cache cannot keep a pack beside a RAM tier this large, and
    // filling the tier through it (page faults on the mapped pack) ran at 0.5 GB/s on a 120 GiB container whose SSD
    // reads 5.2 GB/s with 8 parallel O_DIRECT readers (RTX 3090 + EPYC 7D12 box, 2026-10-07). Slots and expert
    // offsets are 4 KiB aligned in a pack (an expert that is not is copied) and slot lengths 4 KiB multiples. DS41_FILL_DIRECT=0, or a file
    // system that refuses O_DIRECT, copies from the mapped pack as before.
    const char* fill_env = std::getenv("DS41_FILL_DIRECT");
    const int dfd = fill_env && fill_env[0] == '0' ? -1 : open((pack.dir() + "/experts.bin").c_str(), O_RDONLY | O_DIRECT);
    filled_direct_ = dfd >= 0;
    const uint8_t* base = pack.expert_base();
    std::atomic<int> next{0};
    std::atomic<bool> failed{false};
    std::vector<std::thread> pool;
    for (int t = 0; t < (filled_direct_ ? std::max(threads, 16) : std::max(threads, 1)); ++t)
        pool.emplace_back([&] {
            for (int s; (s = next++) < slots_ && !failed;) {
                const ExpertSlot& x = pack.expert(holder_[s].first, holder_[s].second);
                if (filled_direct_ && x.offset % 4096 == 0) {   // an unaligned expert (not in a real pack): copied
                    const size_t len = (size_t) (off_[s + 1] - off_[s]);   // the slot: the expert rounded to 4 KiB
                    size_t got = 0;
                    while (got < x.bytes) {
                        const ssize_t r = pread(dfd, arena_ + off_[s] + got, len - got, (off_t) (x.offset + got));
                        if (r <= 0) break;
                        got += (size_t) r;
                    }
                    if (got < x.bytes) failed = true;
                    continue;
                }
                std::memcpy(arena_ + off_[s], base + x.offset, x.bytes);
                // the file pages are not needed any more: unmap them, so the OS reclaims them first
                const uintptr_t a = ((uintptr_t) (base + x.offset)) & ~(uintptr_t) 4095;
                madvise((void*) a, (uintptr_t) (base + x.offset + x.bytes) - a, MADV_DONTNEED);
            }
        });
    for (auto& t : pool) t.join();
    if (dfd >= 0) close(dfd);
    if (failed) {
        munmap(arena_, arena_bytes_);
        arena_ = nullptr;
        throw std::runtime_error("ds41 RAM tier: a read of " + pack.dir() + "/experts.bin failed");
    }
    // Map the anonymous arena only. Pageable file experts must stay CPU-only.
    if (cudaHostRegister(arena_, arena_bytes_, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
        locked_ = registered_ = true;
        void* alias = nullptr;
        if (cudaHostGetDevicePointer(&alias, arena_, 0) == cudaSuccess)
            device_alias_ = static_cast<uint8_t*>(alias);
        else
            cudaGetLastError();
    } else {
        cudaGetLastError();
        locked_ = mlock(arena_, arena_bytes_) == 0;
    }
    if (!locked_)
        std::fprintf(stderr, "ds41 RAM tier: could not lock %.1f GiB (memlock limit?); it may be paged out\n",
                     arena_bytes_ / 1073741824.0);
    if (!device_alias_)
        std::fprintf(stderr, "ds41 RAM tier: GPU mapping unavailable; RAM experts stay on the CPU\n");
    if (device_alias_) {
        try {
            std::vector<kernels::Exl3Expert> desc((size_t) L * n_experts_);
            for (int s = 0; s < slots_; ++s) {
                const auto [l, e] = holder_[s];
                desc[(size_t) l * n_experts_ + e] =
                    VramExperts::describe_at(pack_, l, e, device_alias_ + off_[s]);
            }
            ck(cudaMalloc(&experts_dev_, desc.size() * sizeof(desc[0])), "descriptor allocation");
            ck(cudaMemcpy(experts_dev_, desc.data(), desc.size() * sizeof(desc[0]), cudaMemcpyHostToDevice),
               "descriptor upload");
            // The next step may use a nonblocking stream.
            ck(cudaStreamSynchronize(nullptr), "descriptor publication");
        } catch (...) {
            cudaFree(experts_dev_);
            cudaHostUnregister(arena_);
            munmap(arena_, arena_bytes_);
            throw;
        }
    }
    for (int s = 0; s < slots_; ++s) {
        publish_residency(slot_[(size_t) holder_[s].first * n_experts_ + holder_[s].second], s);
        point(holder_[s].first, holder_[s].second, slot_ptr(s));
    }
}

HostExperts::~HostExperts() {
    cudaFree(experts_dev_);
    if (!arena_) return;
    if (registered_) cudaHostUnregister(arena_);
    else if (locked_) munlock(arena_, arena_bytes_);
    munmap(arena_, arena_bytes_);
}

void HostExperts::point(int layer, int expert, const uint8_t* bytes) {
    if (handles_.empty()) return;   // no CPU kernel (tests)
    const ExpertSlot& s = pack_.expert(layer, expert);
    auto desc = [&](int c0, int k_tiles, int n_tiles) {
        MoeCpuMatrixDesc d;
        d.trellis = (const uint16_t*) (bytes + s.comp_off[c0]);
        d.suh = (const at::Half*) (bytes + s.comp_off[c0 + 1]);
        d.svh = (const at::Half*) (bytes + s.comp_off[c0 + 2]);
        d.k_tiles = k_tiles;
        d.n_tiles = n_tiles;
        d.tile_w = (int) (s.comp_bytes[c0] / ((uint64_t) k_tiles * n_tiles * 2));
        return d;
    };
    const MoeCpuMatrixDesc g = desc(0, kDim / 16, kMoeInter / 16), u = desc(4, kDim / 16, kMoeInter / 16),
                           d = desc(8, kMoeInter / 16, kDim / 16);
    exl3_moe_cpu_set_expert_raw(handles_[layer], expert, &g, &u, &d, 0);
}

void HostExperts::publish_descriptor(int layer, int expert, int slot) {
    if (!experts_dev_) return;
    kernels::Exl3Expert desc{};
    if (slot >= 0) desc = VramExperts::describe_at(pack_, layer, expert, device_alias_ + off_[slot]);
    ck(cudaMemcpy(experts_dev_ + (size_t) layer * n_experts_ + expert, &desc, sizeof(desc),
                  cudaMemcpyHostToDevice), "descriptor update");
    ck(cudaStreamSynchronize(nullptr), "descriptor publication");
}

void HostExperts::point_to_file(int layer, int expert) {
    publish_descriptor(layer, expert, -1);
    point(layer, expert, pack_.expert_base() + pack_.expert(layer, expert).offset);
    // Read from the file now. Keep its slot reserved until assign().
    publish_residency(slot_[(size_t) layer * n_experts_ + expert], -1);
}

void HostExperts::assign(int slot, int layer, int expert) {
    if (slot < 0 || slot >= slots_) throw std::invalid_argument("HostExperts::assign: bad slot");
    if (pack_.expert(layer, expert).bytes > slot_capacity(slot))
        throw std::invalid_argument("HostExperts::assign: the expert is larger than the slot");
    const auto [ol, oe] = holder_[slot];
    publish_descriptor(ol, oe, -1);
    publish_residency(slot_[(size_t) ol * n_experts_ + oe], -1);
    holder_[slot] = {layer, expert};
    publish_residency(slot_[(size_t) layer * n_experts_ + expert], slot);
    point(layer, expert, slot_ptr(slot));
    publish_descriptor(layer, expert, slot);
}

}  // namespace strata::ds41
