#include "scope/openmp.hpp"

#include "scope/logger.hpp"
#include "scope/numa.hpp"

namespace openmp {

int get_num_threads() {
    return omp_get_num_threads();
}

int get_num_procs() {
    return omp_get_num_procs();
}

void set_num_threads(int nt) {
    return omp_set_num_threads(nt);
}

void set_dynamic(int flag) {
    return omp_set_dynamic(flag);
}

int get_thread_num() {
#if SCOPE_USE_OPENMP == 1
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void set_num_threads_to_numa_allowed_cpus() {
    // get the number of CPUs we are allowed to run on
    auto cpus = numa::get_context_cpus();
    LOG(trace, "NUMA policy: allowed to run on {} CPUs", cpus.size());
    set_num_threads(cpus.size());
}

void init() {
    LOG(debug, "initial configuration:");
    LOG(debug, "openmp num procs   {}", get_num_procs());
    set_dynamic(0);
}

} // namespace openmp


