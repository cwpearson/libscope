#include "lyra/lyra.hpp"

#include "scope/flags.hpp"

namespace scope {


lyra::cli_parser cli;

namespace flags {
// a placeholder so that we have a help string hinting about passthrough to
// google benchmark. we don't consume this
std::string benchmarkStuff;

/*extern*/ bool showHelp = false;
/*extern*/ bool parseError = false;
/*extern*/ std::string parseErrorMessage = "";

} // namespace flags

void parse(int *argc, char **argv) {
  auto result = cli.parse({*argc, argv});
  if (!result) {
      flags::parseError = true;
      flags::parseErrorMessage = result.errorMessage();
  }
}

void show_help(std::ostream &os) {
    os << cli << "\n";
}

void add_flags() {

  cli |= lyra::help(flags::showHelp);
  cli |= lyra::opt(flags::benchmarkStuff,
                   "STUFF")["--benchmark_..."]("passthrough to Google Benchmark");
}


} // namespace scope
