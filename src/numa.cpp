#include <algorithm>
#include <set>
#include <thread>

#if SCOPE_USE_NUMA
#include <numa.h>
#endif

#include "scope/logger.hpp"
#include "scope/numa.hpp"

template<typename T>
void sort_and_uniqify(std::vector<T> &v) {
  std::sort(v.begin(), v.end());
  auto it = std::unique(v.begin(), v.end());
  v.resize(std::distance(v.begin(), it));
}

namespace numa {

void init() {
  if (!available()) {
    return;
  }

#if SCOPE_USE_NUMA == 1
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
#if SCOPE_USE_NUMA == 1
  return -1 != numa_available();
#else
  return false;
#endif
}

void bind_node(const int node) {

#if SCOPE_USE_NUMA == 1
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

int node_count() { return ids().size(); }

std::vector<int> ids() {
  std::vector<int> ret;
#if SCOPE_USE_NUMA

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

std::vector<int> cpu_nodes() {
  return ids();
}

std::vector<int> cpus_in_node(int node) {
#if SCOPE_USE_NUMA
  std::vector<int> ret;
  for (int i = 0; i < numa_num_configured_cpus(); ++i) {
    if (numa_node_of_cpu(i) == node) {
      ret.push_back(i);
    }
  }
  std::sort(ret.begin(), ret.end());
  return ret;
#else
  (void) node;
  std::vector<int> ret;
  for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) {
    ret.push_back(i);
  }
  return ret;
#endif
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