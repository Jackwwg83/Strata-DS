// Design contract only. No implemented GPU kernel is bundled.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace strata_ds {
struct ByteSpan { const std::byte* data; std::size_t size; };
struct ExpertKey { std::uint32_t layer, expert; };
// All EXL3 components, scales, codebooks, rotations and permutations travel together.
struct PackedComponent { std::string name, dtype, layout; ByteSpan bytes; };
struct PackedExpert { ExpertKey key; std::string format, manifest_sha256; std::vector<PackedComponent> parts; };
class Completion {
 public: virtual ~Completion()=default;
 virtual bool query() const=0;
 virtual void wait()=0; // Completion is a device event, not host enqueue success.
};
class ExpertLease {
 public: virtual ~ExpertLease()=default;
 virtual std::uint64_t generation() const=0;
 virtual void release_after(std::shared_ptr<Completion> last_consumer)=0;
};
class ExpertBackend {
 public: virtual ~ExpertBackend()=default;
 virtual std::string capabilities_json() const=0;
 // Source buffers and slots remain leased until the copy AND all consumers complete.
 virtual std::shared_ptr<Completion> stage(const PackedExpert&, ExpertLease&)=0;
 // Refuse unrecognized layouts; never reinterpret EXL3 as GGUF or native FP4.
 virtual void validate(const PackedExpert&) const=0;
};
} // namespace strata_ds
