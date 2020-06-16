#pragma once

#include <spdlog/spdlog.h>

namespace sysbench {
  namespace logger {
    extern std::shared_ptr<spdlog::logger> console;
} // namespace logger
} // namespace sysbench

#define LOG(level, ...) sysbench::logger::console->level(__VA_ARGS__)
