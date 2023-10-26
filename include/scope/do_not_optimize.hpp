#pragma once

namespace scope {
template <class Tp>
__device__ void __attribute__((always_inline))
do_not_optimize(Tp const &value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

// https://gcc.gnu.org/onlinedocs/gcc/Machine-Constraints.html#Machine-Constraints
// https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/2_Cookbook/10_inline_asm
template <class Tp>
__device__ void __attribute__((always_inline)) do_not_optimize(Tp &value) {
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("" : "=v,m"(value) : : "memory");
#else
  asm volatile("" : "+r,m"(value) : : "memory");
#endif
}
} // namespace scope
