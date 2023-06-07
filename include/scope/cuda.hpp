#pragma once

#include <sstream>

#include <cuda_runtime.h>

#include "scope/error.hpp"

cudaError_t cuda_reset_device(const int &id);

namespace scope {
namespace detail {
inline void success_or_throw(cudaError err, const char *file, int line) {
  if (cudaSuccess != err) {
    std::stringstream ss;
    ss << __FILE__ << ":" << __LINE__ << "CUDA error "
       << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
  }
}
} // namespace detail

cudaError hip_reset_device(const int &id);

} // namespace scope

#define CUDA_RUNTIME(x) scope::detail::success_or_throw(x, __FILE__, __LINE__)