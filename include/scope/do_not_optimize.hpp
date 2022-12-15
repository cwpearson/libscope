#pragma once

namespace scope {
template <class Tp>
__device__
void __attribute__((always_inline)) do_not_optimize(Tp const& value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

// https://gcc.gnu.org/onlinedocs/gcc/Machine-Constraints.html#Machine-Constraints
template <class Tp>
__device__
void __attribute__((always_inline)) do_not_optimize(Tp& value) {
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("" : "+v"(value) : : "memory");
#else
  asm volatile("" : "+r,m"(value) : : "memory");
#endif
}
}
