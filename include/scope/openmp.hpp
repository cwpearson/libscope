#pragma once

#if defined(SCOPE_USE_OPENMP) && SCOPE_USE_OPENMP==1
#include <omp.h>
#endif // defined(SCOPE_USE_OPENMP) && SCOPE_USE_OPENMP==1

namespace openmp {

// to be called during scope init
void init();

int get_num_threads();
int get_num_procs();
int get_thread_num();
void set_num_threads(int nt);
void set_dynamic(int flag);


void set_num_threads_to_numa_allowed_cpus();

} // namespace openmp