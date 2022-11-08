#include "scope/device.hpp"

int Device::device_id() const {
    switch (kind_) {
        case Kind::hip: return id_;
        default:
            throw std::runtime_error("can't get device_id for Device kind");
    }
}

bool Device::can_map_host_memory() const {
    switch (kind_) {
    case Kind::hip: return hipDeviceProp_.canMapHostMemory;
    default:
        throw std::runtime_error("can't get can_map_host_memory for Device kind");
    }
}