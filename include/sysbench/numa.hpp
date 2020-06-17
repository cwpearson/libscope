#pragma once

#include <vector>

namespace numa {

/* to be called during sysbench init
*/
void init();

/* True if there is NUMA support and the system supports NUMA, false otherwise
 */
bool available();

/* bind future processing and allocation by this thread to `node`.
If no NUMA support, does nothing
*/
void bind_node(int node);

/* return the number of numa nodes
If no NUMA support, return 1
*/
int node_count();

/* return the NUMA ids present in the system
 */
std::vector<int> ids();

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
