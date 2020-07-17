#include <numeric>

#include "scope/cuda.hpp"
#include "scope/flags.hpp"

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

  /* all GPUs if none specified */
  if (scope::flags::visibleGPUs.empty()) {
    ret.resize(count);
    std::iota(ret.begin(), ret.end(), 0);
    return ret;
  } else { /* one version of each GPU specified */
    for (int id : scope::flags::visibleGPUs) {
      if (id < count && ret.end() == std::find(ret.begin(), ret.end(), id)) {
        ret.push_back(id);
      }
    }
    return ret;
  }
}