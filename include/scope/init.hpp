#pragma once

#include <string>

namespace scope {

void initialize(int *argc, char **argv);
void run();
void finalize();

/* finalizes, then exits with `code`
   should be used instead of exit() after stabilize_system_state() is called
*/
void safe_exit(int code);

// return non-zero if program should exit with that code
typedef int (*InitFn)();

// a function that will run before the InitFns.
// It should register command line options and a version string
typedef void (*BeforeInitFn)();

// A function that will run after InitFns.
// It could be used to programatically register benchmarks
typedef void (*AfterInitFn)();

void RegisterBeforeInit(BeforeInitFn fn);
void RegisterInit(InitFn fn);
AfterInitFn RegisterAfterInit(AfterInitFn fn, const char *name);

// a string that will be returned by later calls to VersionStrings()
void RegisterVersionString(const std::string &s);

const std::vector<std::string> &VersionStrings();

struct InitRegisterer {
  InitRegisterer(InitFn fn) { RegisterInit(fn); }
};

struct BeforeInitRegisterer {
  BeforeInitRegisterer(BeforeInitFn fn) { RegisterBeforeInit(fn); }
};

} // namespace scope

#define SCOPE_REGISTER_BEFORE_INIT(x)                                          \
  static BeforeInitRegisterer _r_before_init_##x(x);

#define SCOPE_REGISTER_INIT(x) static InitRegisterer _r_init_##x(x);

#define SCOPE_CONCAT(a, b) SCOPE_CONCAT2(a, b)
#define SCOPE_CONCAT2(a, b) a##b
#define AFTER_INIT_FN_NAME(x) SCOPE_CONCAT(_after_init_, __LINE__)

#define SCOPE_AFTER_INIT(x, name)                                              \
  static scope::AfterInitFn AFTER_INIT_FN_NAME(x) =                            \
      scope::RegisterAfterInit(x, name);
