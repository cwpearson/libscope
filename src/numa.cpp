#include <set>
#include <algorithm>

#if SYSBENCH_USE_NUMA
#include <numa.h>
#endif

#include "sysbench/numa.hpp"
#include "sysbench/logger.hpp"

namespace numa {

void init() {
  if (!available()) {
    return;
  }

  numa_set_strict(1);
  LOG(debug, "set numa_set_strict(1)");
  numa_set_bind_policy(1);
  LOG(debug, "set numa_set_bind_policy(1)");

  numa_exit_on_warn = 1;
  LOG(debug, "set numa_exit_on_warn = 1");
  numa_exit_on_error = 1;
  LOG(debug, "set numa_exit_on_error = 1");
}

bool available() {
  #if SYSBENCH_USE_NUMA
  return -1 != numa_available();
#else
  return false;
#endif
}

void bind_node(const int node) {

#if SYSBENCH_USE_NUMA
  if (-1 == node) {
    numa_bind(numa_all_nodes_ptr);
  } else if (node >= 0) {
    struct bitmask *nodemask = numa_allocate_nodemask();
    nodemask = numa_bitmask_setbit(nodemask, node);
    numa_bind(nodemask);
    numa_free_nodemask(nodemask);
  } else {
    LOG(critical, "expected node >= -1");
    exit(1);
  }
#else
  (void)node;
#endif
}

int node_count() {
  return ids().size();
}

std::vector<int> ids() {
  std::vector<int> ret;
  #if SYSBENCH_USE_NUMA

  /* we could query numa_bitmask_isbitset for numa_num_possible_nodes(),
  but we only care about NUMA nodes that also have CPUs in them.
  */

  // discover available nodes
  std::set<int> available_nodes;
  for (int i = 0; i < numa_num_configured_cpus(); ++i) {
    available_nodes.insert(numa_node_of_cpu(i));
  }
  for (auto node : available_nodes) {
    ret.push_back(node);
  }

  #else
  ret.push_back(0);
  #endif
  std::sort(ret.begin(), ret.end());
  return ret;
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