#pragma once

#include <spdlog/spdlog.h>

namespace scope {
  namespace logger {
    extern std::shared_ptr<spdlog::logger> console;
} // namespace logger

namespace logging {
  /* set up loggers */
  void init();
}

} // namespace scope

#define LOG(level, ...) scope::logger::console->level(__VA_ARGS__)
