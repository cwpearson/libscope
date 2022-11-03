#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "scope/error.hpp"

/* return the unique CUDA device IDs present on the system
 */
const std::vector<int> unique_cuda_device_ids();

cudaError_t cuda_reset_device(const int &id);
