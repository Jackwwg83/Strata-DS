// src/ds41/expert_stream.cpp - prefill's expert stream. See expert_stream.hpp.
#include "strata/ds41/expert_stream.hpp"

#include "strata/ds41/host_experts.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <stdexcept>

namespace strata::ds41 {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 expert stream: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

ExpertStream::ExpertStream(const Pack& pack, const HostExperts* host, uint8_t* ring, int slots, size_t slot_bytes,
                           int threads)
    : pack_(pack), host_(host), ring_(ring), slots_(slots), slot_bytes_(slot_bytes) {
    if (slots < 1 || threads < 1) throw std::invalid_argument("ds41 expert stream: needs a slot and a reader");
    ck(cudaGetDevice(&device_), "cudaGetDevice");
    const std::string path = pack.dir() + "/experts.bin";
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("ds41 expert stream: cannot open " + path);
    used_.resize(slots);
    for (auto& e : used_) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
    for (int i = 0; i < threads; ++i) threads_.emplace_back([this, i] { reader(i); });
}

ExpertStream::~ExpertStream() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
    for (auto& e : used_) cudaEventDestroy(e);
    if (fd_ >= 0) close(fd_);
}

int64_t ExpertStream::push(const std::vector<std::pair<int, int>>& jobs) {
    std::lock_guard<std::mutex> lk(mu_);
    const int64_t first = (int64_t) jobs_.size();
    jobs_.insert(jobs_.end(), jobs.begin(), jobs.end());
    done_.resize(jobs_.size(), 0);
    cv_.notify_all();
    return first;
}

uint8_t* ExpertStream::wait(int64_t j) {
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(mu_);
    if (j < 0 || j >= (int64_t) jobs_.size()) throw std::logic_error("ds41 expert stream: wait for an unknown job");
    cv_.wait(lk, [&] { return done_[j] || !error_.empty(); });
    if (!error_.empty()) throw std::runtime_error(error_);
    wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return ring_ + (size_t) (j % slots_) * slot_bytes_;
}

void ExpertStream::release(int64_t j, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(mu_);
    if (j != released_) throw std::logic_error("ds41 expert stream: jobs must be released in order");
    ck(cudaEventRecord(used_[j % slots_], stream), "record release");
    ++released_;
    cv_.notify_all();
}

void ExpertStream::drain() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return (released_ == (int64_t) jobs_.size() && next_ == released_) || !error_.empty(); });
    if (!error_.empty()) throw std::runtime_error(error_);
    // every release has an event; the readers hold no job: wait for the device side, then restart the numbering
    for (auto& e : used_) ck(cudaEventSynchronize(e), "drain");
    jobs_.clear();
    done_.clear();
    next_ = released_ = 0;
}

ExpertStream::Stats ExpertStream::take_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s;
    s.from_ram = n_ram_.exchange(0);
    s.from_cache = n_cache_.exchange(0);
    s.from_ssd = n_ssd_.exchange(0);
    s.jobs = s.from_ram + s.from_cache + s.from_ssd;
    s.consumer_wait_ms = wait_ms_;
    wait_ms_ = 0;
    return s;
}

void ExpertStream::reader(int id) {
    (void) id;
    uint8_t* staging = nullptr;
    cudaStream_t st = nullptr;
    std::vector<unsigned char> vec;
    try {
        ck(cudaSetDevice(device_), "cudaSetDevice");
        ck(cudaMallocHost((void**) &staging, slot_bytes_), "pinned staging");
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "reader stream");
        for (;;) {
            int64_t j;
            std::pair<int, int> job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                // job j may start once job j - slots is released (its event then exists)
                cv_.wait(lk, [&] {
                    return stop_ || (next_ < (int64_t) jobs_.size() && next_ - slots_ < released_ && error_.empty());
                });
                if (stop_) break;
                j = next_++;
                job = jobs_[j];
            }
            if (j >= slots_) ck(cudaEventSynchronize(used_[j % slots_]), "wait for the slot");
            const ExpertSlot& x = pack_.expert(job.first, job.second);
            uint8_t* dst = ring_ + (size_t) (j % slots_) * slot_bytes_;
            const int32_t ram = host_ ? host_->slot_of(job.first, job.second) : -1;
            if (ram >= 0) {
                ck(cudaMemcpyAsync(dst, host_->slot_ptr(ram), x.bytes, cudaMemcpyHostToDevice, st), "copy from RAM");
                ++n_ram_;
            } else {
                // was every page cached? (the mapped file shares the page cache with pread)
                const uintptr_t a = (uintptr_t) (pack_.expert_base() + x.offset) & ~(uintptr_t) 4095;
                const size_t len = (uintptr_t) (pack_.expert_base() + x.offset + x.bytes) - a;
                vec.resize((len + 4095) / 4096);
                bool cached = mincore((void*) a, len, vec.data()) == 0;
                for (size_t i = 0; cached && i < vec.size(); ++i) cached = vec[i] & 1;
                uint64_t got = 0;
                while (got < x.bytes) {
                    const ssize_t r = pread(fd_, staging + got, x.bytes - got, (off_t) (x.offset + got));
                    if (r <= 0) throw std::runtime_error("ds41 expert stream: short read of experts.bin");
                    got += (uint64_t) r;
                }
                if (!cached) posix_fadvise(fd_, (off_t) x.offset, (off_t) x.bytes, POSIX_FADV_DONTNEED);
                ck(cudaMemcpyAsync(dst, staging, x.bytes, cudaMemcpyHostToDevice, st), "copy from file");
                ++(cached ? n_cache_ : n_ssd_);
            }
            ck(cudaStreamSynchronize(st), "reader copy");
            {
                std::lock_guard<std::mutex> lk(mu_);
                done_[j] = 1;
            }
            cv_.notify_all();
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(mu_);
        if (error_.empty()) error_ = e.what();
        cv_.notify_all();
    }
    if (st) cudaStreamDestroy(st);
    if (staging) cudaFreeHost(staging);
}

}  // namespace strata::ds41
