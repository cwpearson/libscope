#pragma once

#include "benchmark/benchmark.h"

#if SCOPE_USE_CUDA == 1 && SCOPE_USE_NVTX == 1
#include <nvToolsExt.h>
#endif

#if SCOPE_USE_CUDA == 1
#include "scope/cuda.hpp"
#if __CUDACC__
#include "scope/do_not_optimize.hu"
#endif // __CUDACC__
#endif // SCOPE_USE_CUDA == 1

#if defined(SCOPE_USE_HIP)
#include "scope/hip.hpp"
#if defined(__HIP_DEVICE_COMPILE__)
#include "scope/do_not_optimize.hpp"
#endif // __HIP_DEVICE_COMPILE__
#endif // SCOPE_USE_HIP

#include "scope/barrier.hpp"
#include "scope/benchmark.hpp"
#include "scope/cache.hpp"
#include "scope/chrono.hpp"
#include "scope/config.hpp"
#include "scope/defer.hpp"
#include "scope/error.hpp"
#include "scope/flags.hpp"
#include "scope/governor.hpp"
#include "scope/init.hpp"
#include "scope/logger.hpp"
#include "scope/numa.hpp"
#include "scope/openmp.hpp"
#include "scope/page_size.hpp"
#include "scope/system.hpp"
#include "scope/turbo.hpp"
