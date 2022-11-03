#pragma once

#include <sstream>

#include <hip/hip_runtime.h>

namespace scope {
namespace detail {
inline void success_or_throw(hipError_t err, const char *file, int line) {
    if (hipSuccess != err) {
        std::stringstream ss;
        ss << __FILE__ << ":" << __LINE__ << "HIP error " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}
} // namespace detail

hipError_t hip_reset_device(const int &id);

} // namespace scope


#define HIP_RUNTIME(x) scope::detail::success_or_throw(x, __FILE__, __LINE__)