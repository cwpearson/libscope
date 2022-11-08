#pragma once

#include "scope/hip.hpp"

class Device {
public:

    enum class Kind {
        hip
    };

    Device(const Kind &kind) : kind_(kind) {



    }

    int device_id() const;

    static Device hip_device(int id) {
        Device device(Kind::hip);
        device.id_ = id;
        HIP_RUNTIME(hipGetDeviceProperties(&device.hipDeviceProp_, id));
        return device;
    }

    /*
    HIP: returns hipDeviceProp_t::canMapHostMemory
    */
    bool can_map_host_memory() const;

private:
    Kind kind_;
    int id_;

#if defined(SCOPE_USE_HIP)
    hipDeviceProp_t hipDeviceProp_;
#endif
};

