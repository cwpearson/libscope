#include "scope/hip.hpp"

namespace scope {
hipError_t hip_reset_device(const int &id) {
  hipError_t err = hipSetDevice(id);
  if (err != hipSuccess) {
    return err;
  }
  return hipDeviceReset();
}
} // namespace scope