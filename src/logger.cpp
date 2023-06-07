#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "scope/logger.hpp"

/* extern */ std::shared_ptr<spdlog::logger> scope::logger::console;

namespace scope {

namespace logging {

void init() {
  spdlog::cfg::load_env_levels();
  scope::logger::console = spdlog::stdout_color_mt("scope");
  // spdlog::set_level(spdlog::level::trace);
}

} // namespace logging
} // namespace scope
