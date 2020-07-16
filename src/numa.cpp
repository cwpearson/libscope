#include <algorithm>
#include <map>
#include <set>
#include <thread>

#if SCOPE_USE_NUMA
#include <numa.h>
#endif

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
/* all nodes with CPUs
*/
std::vector<int> nodesWithCPUs;

} // namespace detail

void init() {

  /* cache which nodes have CPUs
  */
#if SCOPE_USE_NUMA
  for (int i = 0; i < numa_num_configured_cpus(); ++i) {
    int node = numa_node_of_cpu(i);
    detail::CPUsInNode[node].push_back(i);
    detail::nodesWithCPUs.push_back(node);
  }
  for (auto &kv : detail::CPUsInNode) {
    std::vector<int> &cpus = kv.second;
    sort_and_uniqify(cpus);
  }
  sort_and_uniqify(detail::nodesWithCPUs);
#else
  (void)node;
  for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) {
    detail::CPUsInNode[0].push_back(i);
  }
  detail::nodesWithCPUs = {0};
#endif

  if (!available()) {
    return;
  }

/* if NUMA is real, then make sure we get what we ask for
*/
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

const std::vector<int> &ids() { return detail::nodesWithCPUs; }

const std::vector<int> &cpu_nodes() { return ids(); }

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