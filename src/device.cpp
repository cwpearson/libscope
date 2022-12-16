#include "scope/device.hpp"

#include <stdexcept>

int Device::device_id() const {
    switch (kind_) {
#if SCOPE_USE_HIP == 1
        case Kind::hip: return id_;
#endif
        default:
            throw std::runtime_error("can't get device_id for Device kind");
    }
}

bool Device::can_map_host_memory() const {
    switch (kind_) {
#if SCOPE_USE_HIP == 1
    case Kind::hip: return hipDeviceProp_.canMapHostMemory;
#endif
    default:
        throw std::runtime_error("can't get can_map_host_memory for Device kind");
    }
}