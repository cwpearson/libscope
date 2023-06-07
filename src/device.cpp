#include "scope/device.hpp"

#include <stdexcept>

#if defined(SCOPE_ENABLE_HIP)
#include "scope/hip.hpp"
#endif

#if defined(SCOPE_ENABLE_CUDA)
#include "scope/cuda.hpp"
#endif

int Device::device_id() const {
    switch (kind_) {
        case Kind::hip: return id_;
        case Kind::cuda: return id_;
        default:
            throw std::runtime_error("can't get device_id for Device kind");
    }
}

bool Device::can_map_host_memory() const {
    switch (kind_) {
    case Kind::hip: 
#if defined(SCOPE_USE_HIP)    
    return hipDeviceProp_.canMapHostMemory;
#endif
    case Kind::cuda:
#if defined(SCOPE_USE_CUDA)
    return cudaDeviceProp_.canMapHostMemory;
#endif
    default:
        throw std::runtime_error("can't get can_map_host_memory for Device kind");
    }
}



#if defined(SCOPE_USE_HIP)
Device::Device::hip_device(int id) {
    Device device(Device::Kind::hip, id);
    HIP_RUNTIME(hipGetDeviceProperties(&device.hipDeviceProp_, id));
    return device;
}
#endif

#if defined(SCOPE_USE_CUDA)
Device Device::cuda_device(int id) {
    Device device(Device::Kind::cuda, id);
    CUDA_RUNTIME(cudaGetDeviceProperties(&device.cudaDeviceProp_, id));
    return device;
}
#endif

