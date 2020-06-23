#pragma once

#include "benchmark/benchmark.h"

#if SCOPE_USE_CUDA == 1 && SCOPE_USE_NVTX == 1
#include <nvToolsExt.h>
#endif

#if SCOPE_USE_CUDA == 1
#include "scope/cuda.hpp"
#endif // SCOPE_USE_CUDA == 1

#include "scope/barrier.hpp"
#include "scope/benchmark.hpp"
#include "scope/cache.hpp"
#include "scope/config.hpp"
#include "scope/defer.hpp"
#include "scope/flags.hpp"
#include "scope/governor.hpp"
#include "scope/init.hpp"
#include "scope/logger.hpp"
#include "scope/numa.hpp"
#include "scope/page_size.hpp"
#include "scope/turbo.hpp"
