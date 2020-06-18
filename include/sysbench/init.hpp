#pragma once

#include <string>

namespace sysbench{ 


void initialize(int *argc, char **argv);
void run();
void finalize();

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

const std::vector<std::string>& VersionStrings();

struct InitRegisterer {
  InitRegisterer(InitFn fn) {
    RegisterInit(fn);
  }
};

struct BeforeInitRegisterer {
  BeforeInitRegisterer(BeforeInitFn fn) {
    RegisterBeforeInit(fn);
  }
};

} // namespace sysbench

#define SYSBENCH_REGISTER_BEFORE_INIT(x) static BeforeInitRegisterer _r_before_init_##x(x);

#define SYSBENCH_REGISTER_INIT(x) static InitRegisterer _r_init_##x(x);

#define SYSBENCH_CONCAT(a, b) SYSBENCH_CONCAT2(a, b)
#define SYSBENCH_CONCAT2(a, b) a##b
#define AFTER_INIT_FN_NAME(x) SYSBENCH_CONCAT(_after_init_, __LINE__)

#define SYSBENCH_AFTER_INIT(x, name) \
  static sysbench::AfterInitFn AFTER_INIT_FN_NAME(x) = sysbench::RegisterAfterInit(x, name);


