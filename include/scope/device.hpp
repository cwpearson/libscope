#pragma once

#if SCOPE_USE_HIP == 1
#include "scope/hip.hpp"
#endif

class Device {
public:

    enum class Kind {
        hip
    };

    Device(const Kind &kind) : kind_(kind) {



    }

    int device_id() const;

#if SCOPE_USE_HIP == 1
    static Device hip_device(int id) {
        Device device(Kind::hip);
        device.id_ = id;
        HIP_RUNTIME(hipGetDeviceProperties(&device.hipDeviceProp_, id));
        return device;
    }
#endif

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

