#include <chrono>

namespace scope {
    using clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<double>;
}