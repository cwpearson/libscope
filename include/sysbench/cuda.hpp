#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "sysbench/error.hpp"

/* return the unique CUDA device IDs present on the system
 */
const std::vector<int> unique_cuda_device_ids();

cudaError_t cuda_reset_device(const int &id);

template <> inline const char *error_string<cudaError_t>(const cudaError_t &status) {
  return cudaGetErrorString(status);
}

template <> inline bool is_success<cudaError_t>(const cudaError_t &err) {
  return err == cudaSuccess;
}