#include "scope/memory_space.hpp"

int MemorySpace::device_id() const {
  switch (kind()) {
  case Kind::hip_device:
  case Kind::hip_mapped_pinned:
  case Kind::cuda_device:
    return deviceId_;
  case Kind::numa:
  case Kind::hip_managed:
  default:
    throw std::runtime_error("can't request device_id from MemorySpace");
  }
}

int MemorySpace::numa_id() const {
  switch (kind()) {
  case Kind::numa:
  case Kind::hip_mapped_pinned:
    return numaId_;
  case Kind::hip_device:
  case Kind::cuda_device:
  case Kind::hip_managed:
  default:
    throw std::runtime_error("can't request numa_id from MemorySpace");
  }
}
