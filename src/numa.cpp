#include <algorithm>
#include <map>
#include <set>
#include <thread>
#include <iostream>

#if defined(SCOPE_USE_NUMA)
#include <numa.h>
#include <unistd.h>
#endif

#include "scope/flags.hpp"
#include "scope/init.hpp"
#include "scope/logger.hpp"
#include "scope/numa.hpp"

template <typename T> void sort_and_uniqify(std::vector<T> &v) {
  std::sort(v.begin(), v.end());
  auto it = std::unique(v.begin(), v.end());
  v.resize(std::distance(v.begin(), it));
}

namespace numa {

namespace detail {

/* cache of node -> cpus
 */
std::map<int, std::vector<int>> CPUsInNode;

/* cache of node -> memory */
std::map<int, std::vector<int>> memoriesInNode;

/* all nodes with CPUs
 */
std::vector<int> nodesWithCPUs;

/* CPUs the process is allowed to run on
*/
std::set<int> allowedCPUs;

/* NUMA nodes the process is allowed to allocate on
*/
std::set<int> allowedMems;

} // namespace detail

void init() {

  /* cache which nodes have CPUs
   */
#if defined(SCOPE_USE_NUMA)

  // determine the CPUs this process is allowed to run
  // this may be quite restricted, especially on
  // managed clusters
  LOG(trace, "numa_num_possible_cpus()={}", numa_num_possible_cpus());
  // struct bitmask *allowedCpus = numa_get_run_node_mask();
  for (int i = 0; i < numa_num_possible_cpus(); ++i) {
    if (numa_bitmask_isbitset(numa_all_cpus_ptr, i)) {
      LOG(trace, "mayrun on CPU {}", i);
      detail::allowedCPUs.insert(i);
    }
  }
  // numa_free_cpumask(allowedCpus);

  // returns a mask of nodes on which the current task
  // is allowed to allocate memory
  struct bitmask *allowedMems = numa_get_mems_allowed();
  for (int i = 0; i < numa_max_possible_node(); ++i) {
    if (numa_bitmask_isbitset(allowedMems, i)) {
      LOG(trace, "may allocate on NUMA node {}", i);
      detail::allowedMems.insert(i);
    }
  }
  numa_free_nodemask(allowedMems);


  for (int cpu : detail::allowedCPUs) {
    int node = numa_node_of_cpu(cpu);
    if (scope::flags::numa_is_visible(node)) {
      detail::CPUsInNode[node].push_back(cpu);
      detail::nodesWithCPUs.push_back(node);
      LOG(trace, "CPU {} in NUMA node {}", cpu, node);
    }
  }
  for (auto &kv : detail::CPUsInNode) {
    std::vector<int> &cpus = kv.second;
    sort_and_uniqify(cpus);
  }
  sort_and_uniqify(detail::nodesWithCPUs);
#else
  if (!scope::flags::visibleNUMAs.empty()) {
    LOG(critical, "NUMA visibility set with --numa, but scope not compiled "
                  "with NUMA support");
    scope::safe_exit(EXIT_FAILURE);
  }
  for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) {
    detail::CPUsInNode[0].push_back(i);
  }
  detail::nodesWithCPUs = {0};
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
    struct bitmask *cpumask = numa_allocate_cpumask();
    numa_node_to_cpus(node, cpumask);
    numa_sched_setaffinity(getpid(), cpumask);
  } else {
    LOG(critical, "expected node >= -1");
    scope::safe_exit(EXIT_FAILURE);
  }
#else
  (void)node;
#endif
}

int node_count() { return ids().size(); }

const std::vector<int> &ids() { return detail::nodesWithCPUs; }

const std::vector<int> &cpu_nodes() { return ids(); }

const std::vector<int> nodes() {
  std::vector<int> ret;
  for (int node : detail::allowedMems) {
    if (scope::flags::numa_is_visible(node)) {
      ret.push_back(node);
    }
  }
  sort_and_uniqify(ret);
  return ret; 
}

std::vector<int> cpus_in_node(int node) {
  if (detail::CPUsInNode.count(node)) {
    return detail::CPUsInNode[node];
  } else {
    return {};
  }
}

std::vector<int> cpus_in_nodes(const std::vector<int> &nodes) {
  std::vector<int> ret;
  for (auto &node : nodes) {
    std::vector<int> cpus = cpus_in_node(node);
    ret.insert(ret.end(), cpus.begin(), cpus.end());
  }
  sort_and_uniqify(ret);
  return ret;
}

bool can_execute_in_node(int node) {
  return !cpus_in_node(node).empty();
}

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



} // namespace numa