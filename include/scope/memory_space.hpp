#pragma once

#include <ostream>

class MemorySpace {
public:
    enum class Kind {
        numa
        ,hip_device
        ,hip_managed
        ,cuda_device
    };

    static MemorySpace numa_space(int id) {
        MemorySpace ms;
        ms.kind_ = Kind::numa;
        ms.numaId_ = id;
        return ms;
    }

#ifdef __HIP__
    static MemorySpace hip_device_space(int id) {
        MemorySpace ms;
        ms.kind_ = Kind::hip_device;
        ms.deviceId_ = id;
        return ms;
    }
    static MemorySpace hip_managed_space() {
        MemorySpace ms;
        ms.kind_ = Kind::hip_managed;
        return ms;
    }
#endif // __HIP__

    const Kind &kind() const noexcept {return kind_;}

    friend std::ostream& operator<<(std::ostream& os, const MemorySpace& ms);

private:
    Kind kind_;
    int deviceId_;
    int numaId_;
};


inline std::ostream& operator<<(std::ostream& os, const MemorySpace& ms)
{
    using Kind = MemorySpace::Kind;

    switch(ms.kind_) {
        case Kind::numa: {
            os << "numa:" << ms.numaId_;
            break;
        }
        case Kind::hip_device: {
            os << "hip_device:" << ms.deviceId_;
            break;
        }
        case Kind::hip_managed: {
            os << "hip_managed";
            break;
        }
        case Kind::cuda_device: {
            os << "cuda_device:" << ms.deviceId_;
            break;
        }
    }
    return os;
}
