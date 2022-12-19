#pragma once

#include <vector>

#if defined(SCOPE_USE_NUMA) && SCOPE_USE_NUMA == 1
#include <numa.h>
#endif

namespace numa {

/* to be called during scope init
*/
void init();

/* True if there is NUMA support and the system supports NUMA, false otherwise
 */
bool available();

/* bind future allocations to this node
   bind execution to legal CPUs in this node
*/
void bind_node(int node);


template <typename T>
T* alloc_node(size_t count, int node) {
#if defined(SCOPE_USE_NUMA) && SCOPE_USE_NUMA == 1
  return static_cast<T*>(numa_alloc_onnode(count * sizeof(T), node));
#else
  return static_cast<T*>(malloc(count * sizeof(T)));
#endif
}

void free_node(void *start, size_t size);

/* return the number of numa nodes
If no NUMA support, return 1
*/
int node_count();

/* return the numa nodes on that we can allocate on
*/
const std::vector<int> nodes();

/* return the NUMA ids that contain CPUs we can execute on
 */
const std::vector<int> &ids();
const std::vector<int> &cpu_nodes();

/* return the CPUs in `node` we can bind to
   if no NUMA support, return all online CPUs
*/
std::vector<int> cpus_in_node(int node);

/* return the CPUs in `nodes` that we can bind to
  if no NUMA support, return the online CPUs
*/
std::vector<int> cpus_in_nodes(const std::vector<int> &nodes);

// true if there is a CPU in node we can execute on
bool can_execute_in_node(int node);

/* bind to `node` while in scope. Release bind on destruction
*/
class ScopedBind {
  bool active;

public:
  ScopedBind(int node);
  ~ScopedBind();
  ScopedBind(const ScopedBind &other) = delete;
  ScopedBind(ScopedBind &&other);
};

} // namespace numa
