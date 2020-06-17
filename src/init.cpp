#include <iostream>

#include <signal.h>

#include "benchmark/benchmark.h"
#include "lyra/lyra.hpp"

#include "sysbench/init.hpp"
#include "sysbench/logger.hpp"
#include "sysbench/flags.hpp"
#include "sysbench/turbo.hpp"


/* handler to restore system turbo state
*/
void handler(int sig) {
  // unregister the handler
  signal(sig, SIG_DFL);
  // restore a previously-recorded turbo state
  turbo::set_state();
  exit(EXIT_FAILURE);
}

namespace sysbench {

typedef struct {
  AfterInitFn fn;
  const char *name;
} AfterInitRecord;

static std::vector<AfterInitRecord> &AfterInits() {
  static std::vector<AfterInitRecord> after_inits;
  return after_inits;
}

static struct { InitFn fn; } inits[5000];
static size_t ninits = 0;

static BeforeInitFn before_inits[5000];
static size_t n_before_inits = 0;

static std::vector<std::string> version_strings;

const std::vector<std::string> &VersionStrings() { return version_strings; }

void do_before_inits() {
  for (size_t i = 0; i < n_before_inits; ++i) {
    before_inits[i]();
  }
}

void do_inits() {
  for (size_t i = 0; i < ninits; ++i) {
    LOG(debug, "Running init function {}", i);
    int status = inits[i].fn();
    if (status) {
      exit(status);
    }
  }
}

void do_after_inits() {
  LOG(debug, "running {} after init functions", AfterInits().size());
  for (size_t i = 0; i < AfterInits().size(); ++i) {
    auto r = AfterInits()[i];
    LOG(debug, "Calling AfterInit  {} ({})", i, r.name);
    r.fn();
    LOG(trace, "Finished AfterInit {} ({})", i, r.name);
  }
}

void initialize(int *argc, char **argv) {

  /* record system turbo state, to be restored in the signal handler*/
  if (turbo::can_modify()) {
    turbo::get_state();
  } else {
    std::cerr << "couldn't control turbo\n";
  }
  signal(SIGINT, handler);

  // have benchmark library consume some flags
  benchmark::Initialize(argc, argv);

  sysbench::add_flags();
  sysbench::parse(argc, argv);

  if (sysbench::flags::parseError) {
    std::cerr << "Error in command line: " << flags::parseErrorMessage
              << std::endl;
    sysbench::show_help(std::cerr);
    exit(EXIT_FAILURE);
  }
  if (sysbench::flags::showHelp) {
    sysbench::show_help(std::cout);
    exit(EXIT_SUCCESS);
  }

  // create logger
  sysbench::logging::init();

  do_before_inits();
  do_inits();
  do_after_inits();
}

void run() { benchmark::RunSpecifiedBenchmarks(); }

void RegisterInit(InitFn fn) {
  if (ninits >= sizeof(inits) / sizeof(inits[0])) {
    LOG(critical, "ERROR: {}@{}: RegisterInit failed, too many inits", __FILE__,
        __LINE__);
    exit(-1);
  }
  inits[ninits].fn = fn;
  ninits++;
}

void RegisterBeforeInit(BeforeInitFn fn) {
  if (n_before_inits >= sizeof(before_inits) / sizeof(before_inits[0])) {
    LOG(critical, "ERROR: {}@{}: RegisterBeforeInit failed, too many functions",
        __FILE__, __LINE__);
    exit(-1);
  }
  before_inits[n_before_inits] = fn;
  n_before_inits++;
}

AfterInitFn RegisterAfterInit(AfterInitFn fn, const char *name) {
  AfterInits().push_back({fn, name});
  return fn;
}

void RegisterVersionString(const std::string &s) {
  version_strings.push_back(s);
}

} // namespace sysbench