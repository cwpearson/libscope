#include <numeric>

#include "scope/cuda.hpp"
#include "scope/flags.hpp"

namespace scope {
cudaError_t cuda_reset_device(const int &id) {
  cudaError_t err = cudaSetDevice(id);
  if (err != cudaSuccess) {
    return err;
  }
  return cudaDeviceReset();
}
} // namespace scope
