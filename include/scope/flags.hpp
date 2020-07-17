/* CLI argument parsing and related values
*/

#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace scope {

namespace flags {
extern bool showHelp; // help was requested on the cli
extern bool parseError; // a cli parse error occured
extern std::string parseErrorMessage; // a description of a cli parse error

extern std::vector<int> visibleGPUs; // GPUs that are visible during benchmark registration
extern std::vector<int> visibleNUMAs; // NUMA nodes visible during benchmark registration
}

/* add the scope flags to the parser
*/
void add_flags();

/* parse the scope flags
*/
void parse(int *argc, char **argv);

/* show a description of the scope flags
*/
void show_help(std::ostream &os);
} // namespace scope