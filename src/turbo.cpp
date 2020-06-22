#include <fstream>

#include <iostream>

#include "sysbench/turbo.hpp"

turbo::State global;

turbo::State::State() : method(Method::NONE) {}

bool can_read_write(const std::string &path) {
  FILE *file = fopen(path.c_str(), "wr");
  if (!file) {
    return false;
  }
  fclose(file);
  return true;
}

bool can_read(const std::string &path) {
  FILE *file = fopen(path.c_str(), "r");
  if (!file) {
    return false;
  }
  fclose(file);
  return true;
}

/* detect the available boost control methods on this system
*/
turbo::Method detect_method() {
  if (can_read("/sys/devices/system/cpu/intel_pstate/no_turbo")) {
    return turbo::Method::PSTATE;
  }
  if (can_read("/sys/devices/system/cpu/cpufreq/boost")) {
    return turbo::Method::CPUFREQ;
  }
  return turbo::Method::NONE;
}

turbo::Result read_one_line(std::string &str, const std::string &path) {
  turbo::Result result = turbo::Result::UNKNOWN;

  FILE *file = fopen(path.c_str(), "r");
  if (!file) {
    if (ENOENT == errno) {
      return turbo::Result::NOT_SUPPORTED;
    } else if (EPERM == errno) {
      return turbo::Result::NO_PERMISSION;
    } else if (EACCES == errno) {
      return turbo::Result::NO_PERMISSION;
    } else {
      return turbo::Result::UNKNOWN;
    }
  }

  char *line = nullptr;
  size_t len;
  size_t read = getline(&line, &len, file);
  if (0 == read) {
    goto finish;
  }
  str = line;
  result = turbo::Result::SUCCESS;

finish:
  free(line);
  fclose(file);
  return result;
}

turbo::Result write_str(const std::string &str, const std::string &path) {
  turbo::Result result = turbo::Result::UNKNOWN;
  FILE *file = fopen(path.c_str(), "w");
  if (!file) {
    if (ENOENT == errno) {
      return turbo::Result::NOT_SUPPORTED;
    } else if (EPERM == errno) {
      return turbo::Result::NO_PERMISSION;
    } else if (EACCES == errno) {
      return turbo::Result::NO_PERMISSION;
    } else {
      std::cerr << errno << "\n";
      return turbo::Result::UNKNOWN;
    }
  }

  size_t written = fwrite(str.c_str(), str.size(), 1, file);
  if (0 == written) {
    result = turbo::Result::UNKNOWN;
  } else {
    result = turbo::Result::SUCCESS;
  }

  fclose(file);
  return result;
}

turbo::Result write_intel_pstate_no_turbo(const std::string &str) {
  return write_str(str, "/sys/devices/system/cpu/intel_pstate/no_turbo");
}

turbo::Result read_intel_pstate_no_turbo(std::string &str) {
  return read_one_line(str, "/sys/devices/system/cpu/intel_pstate/no_turbo");
}

turbo::Result write_acpi_cpufreq_boost(const std::string &str) {
  return write_str(str, "/sys/devices/system/cpu/cpufreq/boost");
}

turbo::Result read_acpi_cpufreq_boost(std::string &str) {
  return read_one_line(str, "/sys/devices/system/cpu/cpufreq/boost");
}

namespace turbo {

bool can_modify() {
  return Method::NONE != detect_method();
}

Result enable() {
  Method method = detect_method();
  if (Method::PSTATE == method) {
    return write_intel_pstate_no_turbo("0");
  } else if (Method::CPUFREQ == method) {
    return write_acpi_cpufreq_boost("1");
  } else {
    return Result::NOT_SUPPORTED;
  }
}

Result disable() {
  Method method = detect_method();
  if (Method::PSTATE == method) {
    return write_intel_pstate_no_turbo("1");
  } else if (Method::CPUFREQ == method) {
    return write_acpi_cpufreq_boost("0");
  } else {
    return Result::NOT_SUPPORTED;
  }
}

Result get_state(State *state) {
  state->method = detect_method();
  Result result;
  
  if (Method::PSTATE == state->method) {
    std::string read;
    result = read_intel_pstate_no_turbo(read);
    if (result == Result::SUCCESS) {
      state->enabled = ("0\n" == read);
    }
  } else if (Method::CPUFREQ== state->method) {
    std::string read;
    result = read_acpi_cpufreq_boost(read);
    if (result == Result::SUCCESS) {
      state->enabled = ("1\n" == read);
    }
  } else if (Method::NONE == state->method) {
    result = Result::NOT_SUPPORTED;
}

  return result;
}

Result set_state(const State &state) {
  if (Method::NONE != state.method) {
    if (state.enabled) {
      return enable();
    } else {
      return disable();
    }
  } else {
    // successfully did nothing
    return Result::SUCCESS;
  }
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

const char *get_string(const Method &method) {
  switch (method) {
  case Method::NONE:
    return "none";
  case Method::CPUFREQ:
    return "cpufreq";
  case Method::PSTATE:
    return "pstate";
  default:
    __builtin_unreachable();
  }
}

Result get_state() { 
    Result ret = get_state(&global); 
    return ret;
    }

Result set_state() { return set_state(global); }

} // namespace turbo
