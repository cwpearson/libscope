#include "scope/governor.hpp"

#include <cassert>
#include <cerrno>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

/*! return a set of CPUs the current thread can run on
 */
std::vector<int> cpus() {
  std::vector<int> result;
  cpu_set_t mask;
  if (sched_getaffinity(0 /*caller*/, sizeof(cpu_set_t), &mask)) {
    assert(0 && "failed sched_getaffinity");
  }
  for (int i = 0; i < CPU_SETSIZE; ++i) {
    if (CPU_ISSET(i, &mask)) {
      result.push_back(i);
    }
  }
  return result;
}

bool can_modify(const int cpu) {
  std::string path("/sys/devices/system/cpu/cpu");
  path += std::to_string(cpu);
  path += "/cpufreq/scaling_governor";
  FILE *file = fopen(path.c_str(), "wr");
  if (!file) {
    return false;
  }
  fclose(file);
  return true;
}

namespace governor {

State global;

Result get_governor(std::string &result, const int cpu) {
  std::string path("/sys/devices/system/cpu/cpu");
  path += std::to_string(cpu);
  path += "/cpufreq/scaling_governor";
  std::ifstream ifs(path, std::ifstream::in);
  std::getline(ifs, result);
  return Result::SUCCESS;
}

Result set_governor(const std::string &governor, const int cpu) {
  std::string path("/sys/devices/system/cpu/cpu");
  path += std::to_string(cpu);
  path += "/cpufreq/scaling_governor";
  std::ofstream ofs(path, std::ofstream::out);
  ofs << governor;
  ofs.close();
  if (ofs.fail()) {
    switch (errno) {
    case EACCES:
      return Result::NO_PERMISSION;
    case ENOENT:
      return Result::NOT_SUPPORTED;
    default:
      return Result::UNKNOWN;
    }
  }
  return Result::SUCCESS;
}

const char *get_string(const Result &result) {
  switch (result) {
  case Result::SUCCESS:
    return "success";
  case Result::NO_PERMISSION:
    return "no permission";
  case Result::NOT_SUPPORTED:
    return "unsupported operation";
  case Result::UNKNOWN:
    return "unknown error";
  default:
    __builtin_unreachable();
  }
}

bool can_modify() {
  bool result = true;
  for (int cpu : cpus()) {
    result &= ::can_modify(cpu);
  }
  return result;
}

Result set_state_maximum() {
  for (int cpu : cpus()) {
    Result res = set_governor("performance", cpu);
    if (Result::SUCCESS != res) {
      return res;
    }
  }
  return Result::SUCCESS;
}

Result set_state_minimum() {
  for (int cpu : cpus()) {
    Result res = set_governor("powersave", cpu);
    if (Result::SUCCESS != res) {
      return res;
    }
  }
  return Result::SUCCESS;
}

Result record() {
  State state;
  for (int cpu : cpus()) {
    std::string gov;
    Result res = get_governor(gov, cpu);
    if (Result::SUCCESS != res) {
      return res;
    }
    state.governors[cpu] = gov;
  }
  global = state;
  return Result::SUCCESS;
}

Result restore() {
  for (auto kv : global.governors) {
    int cpu = kv.first;
    std::string gov = kv.second;
    Result res = set_governor(gov, cpu);
    if (Result::SUCCESS != res) {
      return res;
    }
  }
  return Result::SUCCESS;
}

} // namespace governor