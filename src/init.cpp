#include <iostream>
#include <thread>

#include <signal.h>

#include "benchmark/benchmark.h"
#include "lyra/lyra.hpp"

#include "scope/flags.hpp"
#include "scope/governor.hpp"
#include "scope/init.hpp"
#include "scope/logger.hpp"
#include "scope/numa.hpp"
#include "scope/turbo.hpp"

/* record the state of the system so we can restore when we exit
 */
void record_system_state() {
  {
    turbo::Result res = turbo::get_state();
    if (turbo::Result::SUCCESS != res) {
      LOG(error, "unable to record turbo state");
    } else {
      LOG(info, "recorded system CPU turbo state");
    }
  }
  {
    governor::Result res = governor::record();
    if (governor::Result::SUCCESS != res) {
      LOG(error, "unable to record CPU governor");
    } else {
      LOG(info, "recorded CPU goveror");
    }
  }
}

/* restore the state of the system from `record_system_state()`
 */
void restore_system_state() {
  // restore a previously-recorded turbo state
  {
    turbo::Result res = turbo::set_state();
    if (turbo::Result::SUCCESS != res) {
      LOG(warn,
          "unable to restore turbo state: {}. (use enable-turbo tool if "
          "needed)",
          turbo::get_string(res));
    } else {
      LOG(info, "Restored original turbo state");
    }
  }
  // restore a previously-recorded turbo state
  {
    governor::Result res = governor::restore();
    if (governor::Result::SUCCESS != res) {
      LOG(warn,
          "unable to restore CPU governor: {}. (use set-minimum tool if "
          "needed)",
          governor::get_string(res));
    } else {
      LOG(info, "Restored original CPU governor");
    }
  }
}

void stabilize_system_state() {
  {
    turbo::Result res = turbo::disable();
    if (turbo::Result::SUCCESS != res) {
      LOG(error, "unable to disable CPU turbo: {}. Run with higher privileges?",
      turbo::get_string(res));
    } else {
      LOG(info, "Disabled CPU turbo");
    }
  }
  {
    governor::Result res = governor::set_state_maximum();
    if (governor::Result::SUCCESS != res) {
      LOG(error, "unable to set OS CPU governor to maximum: {}. Run with higher "
                "privileges?", governor::get_string(res));
    } else {
      LOG(info, "Set OS CPU governor to maximum");
    }
  }
  // let CPU frequency ramp
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

/* on SIGINT, we need to restore the system state
 */
void handler(int sig) {
  // unregister the handler
  signal(sig, SIG_DFL);
  restore_system_state();
  raise(sig);
}

namespace scope {

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

  // have benchmark library consume some flags
  benchmark::Initialize(argc, argv);

  scope::add_flags();
  scope::parse(argc, argv);

  if (scope::flags::parseError) {
    std::cerr << "Error in command line: " << flags::parseErrorMessage
              << std::endl;
    scope::show_help(std::cerr);
    exit(EXIT_FAILURE);
  }
  if (scope::flags::showHelp) {
    scope::show_help(std::cout);
    exit(EXIT_SUCCESS);
  }

  // create logger
  scope::logging::init();

  // record the system state and register handler for cleanup,
  // then adjust the system for benchmarking
  signal(SIGINT, handler);
  record_system_state();
  stabilize_system_state();

  numa::init();

  do_before_inits();
  do_inits();
  do_after_inits();
}

void run() { benchmark::RunSpecifiedBenchmarks(); }

void finalize() { restore_system_state(); }

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

} // namespace scope