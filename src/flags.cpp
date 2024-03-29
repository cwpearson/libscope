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
/*extern*/ std::vector<int> visibleGPUs;
/*extern*/ std::vector<int> visibleNUMAs;

bool numa_is_visible(int node) noexcept {
  if (visibleNUMAs.empty()) {
    return true;
  }
  return visibleNUMAs.end() !=
         std::find(visibleNUMAs.begin(), visibleNUMAs.end(), node);
}

bool gpu_is_visible(int id) noexcept {
  if (visibleGPUs.empty()) {
    return true;
  }
  return visibleGPUs.end() !=
         std::find(visibleGPUs.begin(), visibleGPUs.end(), id);
}

} // namespace flags

void parse(int *argc, char **argv) {
  auto result = cli.parse({*argc, argv});
  if (!result) {
    flags::parseError = true;
    flags::parseErrorMessage = result.errorMessage();
  }
}

void show_help(std::ostream &os) { os << cli << "\n"; }

void add_flags() {

  cli |= lyra::help(flags::showHelp);
  cli |= lyra::opt(flags::benchmarkStuff, "STUFF")["--benchmark_..."](
      "passthrough to Google Benchmark");
  cli |= lyra::opt(flags::visibleGPUs, "device ID")["--cuda"](
      "make a GPU visible during benchmark generation");
  cli |= lyra::opt(flags::visibleNUMAs, "NUMA ID")["--numa"](
      "make a NUMA node visible during benchmark generation");
}

} // namespace scope
