#pragma once

namespace scope {
template <class Tp>
__device__
void __attribute__((always_inline)) do_not_optimize(Tp const& value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
__device__
void __attribute__((always_inline)) do_not_optimize(Tp& value) {
  asm volatile("" : "+r,m"(value) : : "memory");
}
}