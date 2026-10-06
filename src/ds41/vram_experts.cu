// src/ds41/vram_experts.cu - see include/strata/ds41/vram_experts.hpp.
#include "strata/ds41/vram_experts.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/host_experts.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace strata::ds41 {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 vram experts: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

std::vector<std::pair<int, int>> read_expert_profile(const std::string& path, int n_layers, int n_experts) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open expert profile " + path);
    const std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    uint32_t h[5];
    if (blob.size() < 24 || blob.compare(0, 4, "STRP") != 0) throw std::runtime_error(path + ": not a Strata profile");
    std::memcpy(h, blob.data() + 4, sizeof h);
    if (h[0] != 1 || (int) h[1] != n_layers || (int) h[2] != n_experts)
        throw std::runtime_error(path + ": profile version " + std::to_string(h[0]) + ", " + std::to_string(h[1]) + "x" +
                                 std::to_string(h[2]) + ", not 1, " + std::to_string(n_layers) + "x" +
                                 std::to_string(n_experts));
    const uint32_t n = h[4];
    if (blob.size() < 24 + (size_t) n * 4) throw std::runtime_error(path + ": profile truncated");
    std::vector<std::pair<int, int>> ranked;
    ranked.reserve(n);
    std::vector<uint8_t> seen((size_t) n_layers * n_experts, 0);
    for (uint32_t i = 0; i < n; ++i) {
        uint16_t p[2];
        std::memcpy(p, blob.data() + 24 + (size_t) i * 4, 4);
        if (p[0] >= n_layers || p[1] >= n_experts || seen[(size_t) p[0] * n_experts + p[1]]++)
            throw std::runtime_error(path + ": bad or repeated pair at rank " + std::to_string(i));
        ranked.emplace_back(p[0], p[1]);
    }
    return ranked;
}

std::vector<ExpertSwap> plan_expert_swaps(const std::vector<float>& usage, const std::vector<int32_t>& res,
                                          int n_layers, int n_experts, int max_swaps) {
    std::vector<ExpertSwap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int l = 0; l < n_layers; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage.data() + (size_t) l * n_experts;
        const int32_t* r = res.data() + (size_t) l * n_experts;
        for (int32_t e = 0; e < n_experts; ++e) {
            if (r[e] < 0) {
                if (u[e] >= 2.0f) cand.emplace_back(u[e], e);
            } else {
                vict.emplace_back(u[e], e);
            }
        }
        if (cand.empty() || vict.empty()) continue;
        // ties: lower expert first, so the plan does not depend on the sort's stability
        std::sort(cand.begin(), cand.end(),
                  [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first || (a.first == b.first && a.second < b.second); });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    std::stable_sort(swaps.begin(), swaps.end(), [](const ExpertSwap& a, const ExpertSwap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) std::max(max_swaps, 0));
    return swaps;
}

VramExperts::VramExperts(const Pack& pack, const std::string& profile_path, int64_t n_slots, size_t reserve_bytes,
                         Adapt adapt)
    : pack_(pack), adapt_(adapt) {
    const int L = pack.n_layers(), E = pack.n_experts();
    ck(cudaGetDevice(&device_), "cudaGetDevice");
    res_host_.assign((size_t) L * E, -1);
    usage_.assign((size_t) L * E, 0.0f);
    ck(cudaMalloc(&res_dev_, res_host_.size() * sizeof(int32_t)), "residency table");
    ck(cudaMalloc(&ws_, kWorkspaceBytes), "K10 workspace");
    ck(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "copy stream");
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e) slot_bytes_ = std::max<size_t>(slot_bytes_, pack.expert(l, e).bytes);
    slot_bytes_ = (slot_bytes_ + 255) / 256 * 256;
    if (n_slots != 0 && !profile_path.empty()) {
        const auto ranked = read_expert_profile(profile_path, L, E);
        if (n_slots < 0) {
            size_t free_b = 0, total_b = 0;
            ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
            const size_t descs = (size_t) L * E * sizeof(kernels::Exl3Expert);
            n_slots = free_b > reserve_bytes + descs ? (int64_t) ((free_b - reserve_bytes - descs) / slot_bytes_) : 0;
        }
        slots_ = (int) std::min<int64_t>(n_slots, (int64_t) ranked.size());
        if (slots_ > 0) {
            ck(cudaMalloc(&arena_, (size_t) slots_ * slot_bytes_), "expert slots");
            desc_host_.resize(slots_);
            const uint8_t* base = pack.expert_base();
            for (int s = 0; s < slots_; ++s) {
                const auto [l, e] = ranked[s];
                const ExpertSlot& x = pack.expert(l, e);
                ck(cudaMemcpy(arena_ + (size_t) s * slot_bytes_, base + x.offset, x.bytes, cudaMemcpyHostToDevice),
                   "expert copy");
                desc_host_[s] = describe(l, e, s);
                res_host_[(size_t) l * E + e] = s;
            }
            ck(cudaMalloc(&experts_dev_, desc_host_.size() * sizeof(desc_host_[0])), "expert descriptors");
            ck(cudaMemcpy(experts_dev_, desc_host_.data(), desc_host_.size() * sizeof(desc_host_[0]),
                          cudaMemcpyHostToDevice),
               "expert descriptors");
        }
    }
    upload_res();
}

VramExperts::~VramExperts() {
    if (copier_.joinable()) copier_.join();
    if (staging_) cudaFreeHost(staging_);
    cudaStreamDestroy(copy_stream_);
    cudaFree(arena_);
    cudaFree(res_dev_);
    cudaFree(experts_dev_);
    cudaFree(ws_);
}

kernels::Exl3Expert VramExperts::describe(int layer, int expert, int slot) const {
    return describe_at(pack_, layer, expert, arena_ + (size_t) slot * slot_bytes_);
}

kernels::Exl3Expert VramExperts::describe_at(const Pack& pack, int layer, int expert, const uint8_t* dst) {
    const ExpertSlot& x = pack.expert(layer, expert);
    auto proj = [&](int c0, int k, int n) {
        kernels::Exl3Proj p;
        p.trellis = (const uint16_t*) (dst + x.comp_off[c0]);
        p.suh = (const __half*) (dst + x.comp_off[c0 + 1]);
        p.svh = (const __half*) (dst + x.comp_off[c0 + 2]);
        p.k = k;
        p.n = n;
        p.tile_w = (int) (x.comp_bytes[c0] / ((uint64_t) (k / 16) * (n / 16) * 2));
        return p;
    };
    kernels::Exl3Expert d;
    d.w1 = proj(0, kDim, kMoeInter);
    d.w3 = proj(4, kDim, kMoeInter);
    d.w2 = proj(8, kMoeInter, kDim);
    return d;
}

void VramExperts::upload_res() {
    ck(cudaMemcpy(res_dev_, res_host_.data(), res_host_.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
       "residency table");
}

void VramExperts::count(const int32_t* routes, int topk) {
    const int L = pack_.n_layers(), E = pack_.n_experts();
    for (int l = 0; l < L; ++l)
        for (int j = 0; j < topk; ++j) {
            const int32_t e = routes[l * topk + j];
            if (e >= 0 && e < E) usage_[(size_t) l * E + e] += 1.0f;
        }
}

void VramExperts::set_host(HostExperts* host) {
    host_ = host;
    if (host_ && host_->slots() > 0 && !staging_)
        ck(cudaMallocHost((void**) &staging_, slot_bytes_), "swap staging");
}

void VramExperts::copy_worker(std::vector<Pending> work) {
    bool ok = cudaSetDevice(device_) == cudaSuccess;
    const uint8_t* base = pack_.expert_base();
    for (size_t i = 0; ok && i < work.size(); ++i) {
        const Pending& w = work[i];
        uint8_t* vslot = arena_ + (size_t) w.vram_slot * slot_bytes_;
        const ExpertSlot& xin = pack_.expert(w.layer, w.in);
        if (w.ram_slot >= 0) {
            // out: VRAM -> staging; in: its RAM slot -> VRAM; then out: staging -> the RAM slot
            const ExpertSlot& xout = pack_.expert(w.layer, w.out);
            uint8_t* rslot = host_->slot_ptr(w.ram_slot);
            ok = cudaMemcpyAsync(staging_, vslot, xout.bytes, cudaMemcpyDeviceToHost, copy_stream_) == cudaSuccess &&
                 cudaMemcpyAsync(vslot, rslot, xin.bytes, cudaMemcpyHostToDevice, copy_stream_) == cudaSuccess &&
                 cudaMemcpyAsync(experts_dev_ + w.vram_slot, &desc_host_[w.vram_slot], sizeof(kernels::Exl3Expert),
                                 cudaMemcpyHostToDevice, copy_stream_) == cudaSuccess &&
                 cudaStreamSynchronize(copy_stream_) == cudaSuccess;
            if (ok) std::memcpy(rslot, staging_, xout.bytes);
        } else {
            // pageable source: the call returns once the bytes are staged; the stream then finishes the DMA
            ok = cudaMemcpyAsync(vslot, base + xin.offset, xin.bytes, cudaMemcpyHostToDevice, copy_stream_) ==
                     cudaSuccess &&
                 cudaMemcpyAsync(experts_dev_ + w.vram_slot, &desc_host_[w.vram_slot], sizeof(kernels::Exl3Expert),
                                 cudaMemcpyHostToDevice, copy_stream_) == cudaSuccess;
        }
    }
    ok = ok && cudaStreamSynchronize(copy_stream_) == cudaSuccess;
    copy_error_ = !ok;
    copies_done_.store(true, std::memory_order_release);
}

int VramExperts::commit_pending(bool wait) {
    if (pending_.empty()) return 0;
    if (!wait && !copies_done_.load(std::memory_order_acquire)) return 0;   // still copying: the next step checks again
    const int E = pack_.n_experts();
    copier_.join();
    if (copy_error_) throw std::runtime_error("ds41 vram experts: an adaptive expert copy failed");
    for (const Pending& w : pending_) {
        res_host_[(size_t) w.layer * E + w.in] = w.vram_slot;
        if (w.ram_slot >= 0) host_->assign(w.ram_slot, w.layer, w.out);   // the CPU now reads `out` from RAM
    }
    const int committed = (int) pending_.size();
    swaps_total_ += committed;
    pending_.clear();
    upload_res();
    return committed;
}

int VramExperts::between_steps() {
    if (slots_ == 0 || adapt_.every <= 0) return 0;
    if (lent_) throw std::logic_error("ds41 vram experts: a step while slots are lent to prefill");
    const int L = pack_.n_layers(), E = pack_.n_experts();
    const bool was_pending = !pending_.empty();
    const int committed = commit_pending(false);
    if (was_pending && !pending_.empty()) return 0;   // still copying
    if (++calls_ % adapt_.every != 0) return committed;
    const auto swaps = plan_expert_swaps(usage_, res_host_, L, E, adapt_.max_swaps);
    for (float& v : usage_) v *= adapt_.decay;
    if (swaps.empty()) return committed;
    for (const ExpertSwap& s : swaps) {
        const size_t out = (size_t) s.layer * E + s.out;
        const int32_t slot = res_host_[out];
        res_host_[out] = -1;   // evicted now: the CPU computes it (from the file) from the next step on
        desc_host_[slot] = describe(s.layer, s.in, slot);
        const int32_t ram = host_ ? host_->slot_of(s.layer, s.in) : -1;
        if (ram >= 0) host_->point_to_file(s.layer, s.in);   // its RAM slot is about to be overwritten
        pending_.push_back(Pending{s.layer, s.in, s.out, slot, ram});
    }
    upload_res();   // before the copies start: no step reads a slot that is being overwritten
    copies_done_.store(false, std::memory_order_relaxed);
    copier_ = std::thread(&VramExperts::copy_worker, this, pending_);
    return committed;
}

uint8_t* VramExperts::lend(int n) {
    if (lent_) throw std::logic_error("ds41 vram experts: slots already lent");
    if (n < 0 || n > slots_) throw std::invalid_argument("ds41 vram experts: cannot lend that many slots");
    commit_pending(true);
    const int E = pack_.n_experts();
    lent_owner_.assign(n, {-1, -1});
    for (size_t i = 0; i < res_host_.size(); ++i) {
        const int32_t s = res_host_[i];
        if (s >= slots_ - n) {
            lent_owner_[s - (slots_ - n)] = {(int) (i / E), (int) (i % E)};
            res_host_[i] = -1;
        }
    }
    lent_ = n;
    upload_res();
    return arena_ + (size_t) (slots_ - n) * slot_bytes_;
}

void VramExperts::restore() {
    if (!lent_) return;
    const int E = pack_.n_experts();
    const uint8_t* base = pack_.expert_base();
    for (int i = 0; i < lent_; ++i) {
        const auto [l, e] = lent_owner_[i];
        if (l < 0) continue;
        const int s = slots_ - lent_ + i;
        const ExpertSlot& x = pack_.expert(l, e);
        ck(cudaMemcpy(arena_ + (size_t) s * slot_bytes_, base + x.offset, x.bytes, cudaMemcpyHostToDevice),
           "restore lent slot");
        res_host_[(size_t) l * E + e] = s;   // desc_host_ and experts_dev_ still describe (l, e) at slot s
    }
    lent_ = 0;
    lent_owner_.clear();
    upload_res();
}

}  // namespace strata::ds41
