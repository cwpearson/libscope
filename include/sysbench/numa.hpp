#pragma once

#include <vector>

namespace numa {

/* true if system supports NUMA, false otherwise
 */
bool available();

/* bind processing and allocation to `node`.
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
