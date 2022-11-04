#pragma once

#include <vector>

#include "scope/memory_space.hpp"
#include "scope/numa.hpp"

#if defined(SCOPE_USE_HIP)
#include "scope/hip.hpp"
#endif

namespace scope {
namespace system {


std::vector<MemorySpace> hip_memory_spaces();
std::vector<MemorySpace> numa_memory_spaces();
std::vector<MemorySpace> memory_spaces(const std::vector<MemorySpace::Kind> &kinds = {});
inline std::vector<MemorySpace> memory_spaces(const MemorySpace::Kind &kind) {
    std::vector<MemorySpace::Kind> kinds;
    kinds.push_back(kind);
    return memory_spaces(kinds);
}


enum class TransferMethod {
    hip_memcpy,
    hip_memcpy_async,
};

std::vector<TransferMethod> transfer_methods(const MemorySpace &src, const MemorySpace &dst);

/* return HIP device IDs that the process can execute on
*/
std::vector<int> hip_device_ids();

} // namespace system
} // namespace scope

std::ostream& operator<<(std::ostream& os, const scope::system::TransferMethod& tm);