#pragma once

#include <ostream>

class MemorySpace {
public:
  enum class Kind {
    numa,
    hip_device,
    hip_managed,
    hip_mapped_pinned,
    cuda_device,
    cuda_managed,
    cuda_pinned
  };

  static MemorySpace numa_space(int id) {
    MemorySpace ms;
    ms.kind_ = Kind::numa;
    ms.numaId_ = id;
    return ms;
  }

  static MemorySpace hip_device_space(int id) {
    MemorySpace ms;
    ms.kind_ = Kind::hip_device;
    ms.deviceId_ = id;
    return ms;
  }

  static MemorySpace hip_managed_space(int id) {
    MemorySpace ms;
    ms.kind_ = Kind::hip_managed;
    ms.numaId_ = id;
    return ms;
  }

  static MemorySpace hip_mapped_pinned(int deviceId, int numaId) {
    MemorySpace ms;
    ms.kind_ = Kind::hip_mapped_pinned;
    ms.deviceId_ = deviceId;
    ms.numaId_ = numaId;
    return ms;
  }

  static MemorySpace cuda_device_space(int id) {
    MemorySpace ms;
    ms.kind_ = Kind::cuda_device;
    ms.deviceId_ = id;
    return ms;
  }

  static MemorySpace cuda_managed_space(int id) {
    MemorySpace ms;
    ms.kind_ = Kind::cuda_managed;
    ms.numaId_ = id;
    return ms;
  }

  static MemorySpace cuda_pinned(int deviceId, int numaId) {
    MemorySpace ms;
    ms.kind_ = Kind::cuda_pinned;
    ms.deviceId_ = deviceId;
    ms.numaId_ = numaId;
    return ms;
  }

  const Kind &kind() const noexcept { return kind_; }
  int device_id() const;
  int numa_id() const;

  friend std::ostream &operator<<(std::ostream &os, const MemorySpace &ms);

private:
  Kind kind_;
  int deviceId_;
  int numaId_;
};

inline std::ostream &operator<<(std::ostream &os, const MemorySpace &ms) {
  using Kind = MemorySpace::Kind;

  switch (ms.kind_) {
  case Kind::numa: {
    os << "numa:" << ms.numaId_;
    break;
  }
  case Kind::hip_device: {
    os << "hip_device:" << ms.deviceId_;
    break;
  }
  case Kind::hip_managed: {
    os << "hip_managed:" << ms.numaId_;
    break;
  }
  case Kind::hip_mapped_pinned: {
    os << "hip_mapped_pinned:" << ms.deviceId_ << ":" << ms.numaId_;
    break;
  }
  case Kind::cuda_device: {
    os << "cuda_device:" << ms.deviceId_;
    break;
  }
  }
  return os;
}
