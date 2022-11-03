#pragma once

#include <vector>

#include "scope/memory_space.hpp"
#include "scope/numa.hpp"

#ifdef __HIP__
#include "scope/hip.hpp"
#endif

namespace scope {
namespace system {

inline std::vector<MemorySpace> hip_memory_spaces() {
    int ndev;
    HIP_RUNTIME(hipGetDeviceCount(&ndev));

    std::vector<MemorySpace> ret;

    for (int i = 0; i < ndev; ++i) {
        ret.push_back(MemorySpace::hip_device_space(i));
    }
    ret.push_back(MemorySpace::hip_managed_space());
    return ret;
}

inline std::vector<MemorySpace> numa_memory_spaces() {

    auto nodes = numa::nodes();

    std::vector<MemorySpace> ret;
    for (const auto &node : nodes) {
        ret.push_back(MemorySpace::numa_space(node));
    }

    return ret;
}


std::vector<MemorySpace> memory_spaces() {

    std::vector<MemorySpace> spaces;

    auto numaSpaces = numa_memory_spaces();
    spaces.insert(spaces.begin(), numaSpaces.begin(), numaSpaces.end());

#ifdef __HIP__
    auto hipSpaces = hip_memory_spaces();
    spaces.insert(spaces.begin(), hipSpaces.begin(), hipSpaces.end());
#endif

    return spaces;

};

enum class TransferMethod {
    hip_memcpy,
    hip_memcpy_async,
};

std::vector<TransferMethod> transfer_methods(const MemorySpace &src, const MemorySpace &dst) {
    
    using Kind = MemorySpace::Kind;
    
    switch(src.kind()) {
        case Kind::hip_device: {
            switch(dst.kind()) {
                case Kind::hip_device:
                    return {TransferMethod::hip_memcpy, TransferMethod::hip_memcpy_async};
                case Kind::hip_managed:
                    return {TransferMethod::hip_memcpy, TransferMethod::hip_memcpy_async};
                case Kind::numa:
                    return {TransferMethod::hip_memcpy, TransferMethod::hip_memcpy_async};
                case Kind::cuda_device:
                    return {};
            }
        }
        case Kind::hip_managed: {
            switch(dst.kind()) {
                case Kind::hip_device:
                    return {TransferMethod::hip_memcpy, TransferMethod::hip_memcpy_async};
                case Kind::hip_managed:
                    return {TransferMethod::hip_memcpy, TransferMethod::hip_memcpy_async};
                case Kind::numa:
                    return {TransferMethod::hip_memcpy, TransferMethod::hip_memcpy_async};
                case Kind::cuda_device:
                    throw std::runtime_error("unimplemented");
            }
        }
        case Kind::numa: {
            switch(dst.kind()) {
                case Kind::hip_device:
                    return {TransferMethod::hip_memcpy, TransferMethod::hip_memcpy_async};
                case Kind::hip_managed:
                    return {TransferMethod::hip_memcpy, TransferMethod::hip_memcpy_async};
                case Kind::numa:
                    return {TransferMethod::hip_memcpy, TransferMethod::hip_memcpy_async};
                case Kind::cuda_device:
                    throw std::runtime_error("unimplemented");
            }
        }
        case Kind::cuda_device: {
            throw std::runtime_error("unimplemented");
        }
    }

    return {};
}

} // namespace system
} // namespace scope

std::ostream& operator<<(std::ostream& os, const scope::system::TransferMethod& tm)
{
    using TransferMethod = scope::system::TransferMethod;

    switch(tm) {
        case TransferMethod::hip_memcpy: {
            os << "hipMemcpy";
            break;
        }
        case TransferMethod::hip_memcpy_async: {
            os << "hipMemcpyAsync";
            break;
        }
    }
    return os;
}