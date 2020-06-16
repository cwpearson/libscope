#if SYSBENCH_USE_NUMA
#include <numa.h>
#endif

#include "sysbench/numa.hpp"
#include "sysbench/logger.hpp"

namespace numa {

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

    for(int i = 0; i < numa_num_possible_nodes(); ++i) {
      if (numa_bitmask_isbitset(numa_all_nodes_ptr, i)) {
        ret.push_back(i);
      }
    }

  #else
  ret.push_back(0);
  #endif
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