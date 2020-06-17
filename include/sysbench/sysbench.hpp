#pragma once

#include "benchmark/benchmark.h"

#if SYSBENCH_USE_CUDA == 1 && SYSBENCH_USE_NVTX == 1
#include <nvToolsExt.h>
#endif

#if SYSBENCH_USE_CUDA == 1
#include "sysbench/cuda.hpp"
#endif // SYSBENCH_USE_CUDA == 1


#include "sysbench/turbo.hpp"
#include "sysbench/benchmark.hpp"
#include "sysbench/cache.hpp"
#include "sysbench/defer.hpp"
#include "sysbench/flags.hpp"
#include "sysbench/init.hpp"
#include "sysbench/logger.hpp"
#include "sysbench/numa.hpp"
#include "sysbench/page_size.hpp"
#include "sysbench/config.hpp"
