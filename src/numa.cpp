#include <algorithm>
#include <set>
#include <thread>

#if defined(SCOPE_USE_NUMA) && SCOPE_USE_NUMA == 1
#include <numa.h>
#include <unistd.h>
#endif

#include "scope/flags.hpp"
#include "scope/init.hpp"
#include "scope/logger.hpp"
#include "scope/numa.hpp"

namespace numa {

#if defined(SCOPE_USE_NUMA)
inline std::set<int> cpumask_to_set(const struct bitmask *mask) {
  std::set<int> cpus;
  for (int i = 0; i < numa_num_possible_cpus(); ++i) {
    if (numa_bitmask_isbitset(mask, i)) {
      cpus.insert(i);
    }
  }
  return cpus;
}
#endif

#if defined(SCOPE_USE_NUMA)
inline std::set<int> nodemask_to_set(const struct bitmask *mask) {
  std::set<int> nodes;
  for (int i = 0; i < numa_num_possible_nodes(); ++i) {
    if (numa_bitmask_isbitset(mask, i)) {
      nodes.insert(i);
    }
  }
  return nodes;
}
#endif

inline std::set<int> all_cpus() {
#if defined(SCOPE_USE_NUMA)
  // points to a bitmask that is allocated by the library with
  // bits representing all cpus on which the calling task may
  // execute
  return cpumask_to_set(numa_all_cpus_ptr);
#else
  std::set<int> cpus;
  for (size_t i = 0; i < std::thread::hardware_concurrency(); ++i) {
    cpus.insert(i);
  }
  return cpus;
#endif
}

inline std::set<int> all_mems() {
#if defined(SCOPE_USE_NUMA)
  // points to a bitmask that is allocated by the library with
  // bits representing all nodes on which the calling task may
  // allocate memory
  return nodemask_to_set(numa_all_nodes_ptr);
#else
  return {0};
#endif
}

void init() {

  /* cache which nodes have CPUs
   */
#if defined(SCOPE_USE_NUMA)

  // determine the CPUs this process is allowed to run
  // this may be quite restricted, especially on
  // managed clusters
  LOG(trace, "numa_num_possible_cpus()={}", numa_num_possible_cpus());
  // struct bitmask *allowedCpus = numa_get_run_node_mask();
  for (int cpu : all_cpus()) {
    LOG(debug, "may run on CPU {}", cpu);
  }

  for (int mem : all_mems()) {
    LOG(debug, "may allocate on node {}", mem);
  }

  for (int cpu : all_cpus()) {
    int node = numa_node_of_cpu(cpu);
    if (scope::flags::numa_is_visible(node)) {
      LOG(debug, "CPU {} in NUMA node {}", cpu, node);
    }
  }
#else
  if (!scope::flags::visibleNUMAs.empty()) {
    LOG(critical, "NUMA visibility set with --numa, but scope not compiled "
                  "with NUMA support");
    scope::safe_exit(EXIT_FAILURE);
  }
#endif
  if (!available()) {
    if (!scope::flags::visibleNUMAs.empty()) {
      LOG(critical, "NUMA visibility set with --numa, but NUMA not available");
      scope::safe_exit(EXIT_FAILURE);
    }
    return;
  }

/* if NUMA is real, then make sure we get what we ask for
 */
#if defined(SCOPE_USE_NUMA)
  numa_set_strict(1);
  LOG(debug, "set numa_set_strict(1)");
  numa_set_bind_policy(1);
  LOG(debug, "set numa_set_bind_policy(1)");

  numa_exit_on_warn = 1;
  LOG(debug, "set numa_exit_on_warn = 1");
  numa_exit_on_error = 1;
  LOG(debug, "set numa_exit_on_error = 1");
#endif
}

bool available() {
#if defined(SCOPE_USE_NUMA)
  return -1 != numa_available();
#else
  return false;
#endif
}

void bind_node(const int node) {

  LOG(trace, "numa::bind_node({})", node);

#if defined(SCOPE_USE_NUMA)
  if (-1 == node) {
    numa_bind(numa_all_nodes_ptr);
  } else if (node >= 0) {
    // bind allocations
    struct bitmask *nodemask = numa_allocate_nodemask();
    nodemask = numa_bitmask_setbit(nodemask, node);
    LOG(trace, "numa_set_membind(...)", node);
    numa_set_membind(nodemask);

    // bind execution
    numa_run_on_node(node);
  } else {
    LOG(critical, "expected node >= -1");
    scope::safe_exit(EXIT_FAILURE);
  }
#else
  (void)node;
#endif
}

std::set<int> mems() {
  std::set<int> ret;
#if defined(SCOPE_USE_NUMA)
  for (int node : all_mems()) {
    if (scope::flags::numa_is_visible(node)) {
      ret.insert(node);
    }
  }
#else
  ret.insert(0);
#endif
  return ret;
}

std::set<int> cpus_in_node(int node) {
  std::set<int> cpus;
#if defined(SCOPE_USE_NUMA)
  if (scope::flags::numa_is_visible(node)) {
    for (int cpu : all_cpus()) {
      if (node == numa_node_of_cpu(cpu)) {
        cpus.insert(cpu);
      }
    }
  }
#else
  for (int i = 0; i < std::thread::hardware_concurrency(); ++i) {
    cpus.insert(i);
  }
#endif
  return cpus;
}

std::set<int> cpus_in_nodes(const std::set<int> &nodes) {
  std::set<int> ret;
  for (int node : nodes) {
    std::set<int> cpus = cpus_in_node(node);
    ret.insert(cpus.begin(), cpus.end());
  }
  return ret;
}

bool can_execute_in_node(int node) { return !cpus_in_node(node).empty(); }

void bind_cpu(const std::vector<int> &cpus) {
  // allocate CPU maks
#if defined(SCOPE_USE_NUMA)
  struct bitmask *mask = numa_allocate_cpumask();

  for (int cpu : cpus) {
    numa_bitmask_setbit(mask, cpu);
  }

  // set bits
  numa_run_on_node_mask(mask);

  numa_free_cpumask(mask);
#endif
}

void *alloc_onnode(size_t size, int node) {
#if defined(SCOPE_USE_NUMA)
  return numa_alloc_onnode(size, node);
#else
  (void) node;
  return malloc(size);
#endif
}

void free(void *start, size_t size) { 
#if defined(SCOPE_USE_NUMA)
  return numa_free(start, size); 
#else
  (void) size;
  ::free(start);
#endif
  }

ScopedBind::ScopedBind(int node) : active(true) { bind_node(node); }

ScopedBind::~ScopedBind() {
  if (active) {
    bind_node(-1);
  }
}
ScopedBind::ScopedBind(ScopedBind &&other) {
  active = other.active;
  other.active = false;
}

std::set<int> get_context_cpus() {
#if defined(SCOPE_USE_NUMA)
  struct bitmask *affinity = numa_allocate_cpumask();
  numa_sched_getaffinity(getpid(), affinity);
  auto cpus = cpumask_to_set(affinity);
  numa_free_cpumask(affinity);
  return cpus;
#else
  return all_cpus();
#endif
}

std::set<int> get_context_mems() {
#if defined(SCOPE_USE_NUMA)
  // returns the mask of nodes from which the process is allowed to
  // allocate memory in it's current cpuset context
  struct bitmask *nodeMask = numa_get_mems_allowed();
  return nodemask_to_set(nodeMask);
#else
  return all_mems();
#endif
}

} // namespace numa