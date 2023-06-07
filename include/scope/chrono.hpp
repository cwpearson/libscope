#include <chrono>

namespace scope {
using clock = std::chrono::high_resolution_clock;
using duration = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clock>;
} // namespace scope