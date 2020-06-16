#include <spdlog/sinks/stdout_color_sinks.h>

#include "sysbench/logger.hpp"

/* extern */ std::shared_ptr<spdlog::logger> sysbench::logger::console;

namespace sysbench {

namespace logging {

void init() { sysbench::logger::console = spdlog::stdout_color_mt("scope"); }

} // namespace logging
} // namespace sysbench
