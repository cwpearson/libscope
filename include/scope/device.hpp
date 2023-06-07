#pragma once

#if defined(SCOPE_USE_HIP)
#include "scope/hip.hpp"
#endif

#if defined(SCOPE_USE_CUDA)
#include "scope/cuda.hpp"
#endif

class Device {
public:
  enum class Kind { cuda, hip };

  Device(const Kind &kind, int id) : kind_(kind), id_(id) {}

  int device_id() const;

  operator int() const { return id_; }

  /*
  HIP: returns hipDeviceProp_t::canMapHostMemory
  */
  bool can_map_host_memory() const;

#if defined(SCOPE_USE_HIP)
  static Device hip_device(int id);
#endif

#if defined(SCOPE_USE_CUDA)
  static Device cuda_device(int id);
#endif

private:
  Kind kind_;
  int id_;

#if defined(SCOPE_USE_HIP)
  hipDeviceProp_t hipDeviceProp_;
#endif

#if defined(SCOPE_USE_CUDA)
  cudaDeviceProp cudaDeviceProp_;
#endif
};
