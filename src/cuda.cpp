#include <numeric>

#include "sysbench/cuda.hpp"

cudaError_t cuda_reset_device(const int &id) {
  cudaError_t err = cudaSetDevice(id);
  if (err != cudaSuccess) {
    return err;
  }
  return cudaDeviceReset();
}

const std::vector<int> unique_cuda_device_ids() {
  std::vector<int> ret;
  int count;
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&count))) {
    return ret;
  }
  ret.resize(count);
  std::iota(ret.begin(), ret.end(), 0);
  return ret;
}