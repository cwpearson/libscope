#include "scope/system.hpp"
#include "scope/logger.hpp"
#include "scope/flags.hpp"

namespace scope {
namespace system {


std::vector<MemorySpace> hip_memory_spaces() {
    std::vector<MemorySpace> ret;

    const std::vector<Device> hips = hip_devices();
    for (const Device &hip : hips) {
        ret.push_back(MemorySpace::hip_device_space(hip.device_id()));
    }

    const auto numaIds = numa::mems();
    for (const auto &numaId : numaIds) {
        for (const Device &hip : hips) {
            ret.push_back(MemorySpace::hip_mapped_pinned(hip.device_id(), numaId));
        }
    }

    for (const auto &numaId : numaIds) {
        ret.push_back(MemorySpace::hip_managed_space(numaId));
    }
    return ret;
}


std::vector<MemorySpace> numa_memory_spaces() {
    auto nodes = numa::mems();

    std::vector<MemorySpace> ret;
    for (const auto &node : nodes) {
        ret.push_back(MemorySpace::numa_space(node));
    }

    return ret;
}

std::vector<MemorySpace> memory_spaces(const std::vector<MemorySpace::Kind> &kinds) {
    LOG(trace, "scope::system::memory_spaces()...");

    std::vector<MemorySpace> spaces;

    auto numaSpaces = numa_memory_spaces();
    LOG(trace, "scope::system::memory_spaces(): {} numa spaces", numaSpaces.size());
    spaces.insert(spaces.begin(), numaSpaces.begin(), numaSpaces.end());

    auto hipSpaces = hip_memory_spaces();
    LOG(trace, "scope::system::memory_spaces(): {} HIP spaces", hipSpaces.size());
    spaces.insert(spaces.begin(), hipSpaces.begin(), hipSpaces.end());

    // if kinds are provided, only keep matches
    if (!kinds.empty()) {
        std::vector<MemorySpace> ret;
        std::copy_if(spaces.begin(), spaces.end(), std::back_inserter(ret),
        [&](const MemorySpace &ms){
            for (const MemorySpace::Kind &kind : kinds) {
                if (ms.kind() == kind) {
                    return true;
                }
            }
            return false;
        });
        return ret;
    } else {
        return spaces;
    }
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

std::vector<Device> hip_devices() {
    std::vector<Device> ret;
#if defined(SCOPE_USE_HIP)
    int ndev;
    HIP_RUNTIME(hipGetDeviceCount(&ndev));
    for (int i = 0; i < ndev; ++i) {
        if (scope::flags::gpu_is_visible(i)) {
             ret.push_back(Device::hip_device(i));
        }
    }
#endif
    return ret;
}

std::vector<Device> cuda_devices() {
    std::vector<Device> ret;
#if defined(SCOPE_USE_CUDA)
    int ndev;
    CUDA_RUNTIME(cudaGetDeviceCount(&ndev));
    for (int i = 0; i < ndev; ++i) {
        if (scope::flags::gpu_is_visible(i)) {
             ret.push_back(Device::cuda_device(i));
        }
    }
#endif
    return ret;
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