#pragma once

namespace scope {

__host__ inline __attribute__((always_inline)) void clobber_memory() {
  std::atomic_signal_fence(std::memory_order_acq_rel);
}

__device__ inline __attribute__((always_inline)) void clobber_memory() {
  asm volatile("" : : : "memory");
}

} // namespace scope