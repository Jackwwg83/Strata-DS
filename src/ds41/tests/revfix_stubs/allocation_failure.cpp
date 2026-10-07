// Fail one C++ allocation on the calling thread. Worker allocations are unaffected.
#include <cstdlib>
#include <new>
thread_local int revfix_allocation_fail = -1;
void* operator new(std::size_t size) {
    if (revfix_allocation_fail >= 0 && revfix_allocation_fail-- == 0) throw std::bad_alloc();
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
