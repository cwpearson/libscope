#include <spdlog/sinks/stdout_color_sinks.h>

#include "sysbench/logger.hpp"

std::shared_ptr<spdlog::logger> sysbench::logger::console = spdlog::stdout_color_mt("scope");
