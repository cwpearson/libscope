#include "scope/do_not_optimize.hu"

template<> __device__ void do_not_optimize<int32_t>(const int32_t& t) {
    asm volatile("" ::"r"(t) : "memory");
  }
  
  template<> __device__ void do_not_optimize<int64_t>(const int64_t& t) {
    asm volatile("" ::"l"(t) : "memory");
  }