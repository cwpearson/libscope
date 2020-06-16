/* CLI argument parsing and related values
*/

#pragma once

#include <ostream>
#include <string>

namespace sysbench {

namespace flags {
extern bool showHelp; // help was requested on the cli
extern bool parseError; // a cli parse error occured
extern std::string parseErrorMessage; // a description of a cli parse error
}

/* add the sysbench flags to the parser
*/
void add_flags();

/* parse the sysbench flags
*/
void parse(int *argc, char **argv);

/* show a description of the sysbench flags
*/
void show_help(std::ostream &os);
} // namespace sysbench