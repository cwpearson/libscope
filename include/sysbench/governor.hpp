#pragma once

#include <map>

namespace governor {


enum class Result {
  NO_PERMISSION,
  NOT_SUPPORTED,
  SUCCESS,
  UNKNOWN
};

const char *get_string(const Result &result);

/* the governor for each CPU
*/
struct State {
#ifdef __linux__
    std::map<int, std::string> governors;
#else
#error "unsupported platform"
#endif
};

/* whether modifying the governor is supported
*/
bool can_modify();

/* "performance" on linux
*/
Result set_state_maximum();

/* "powersave" on linux
*/
Result set_state_minimum();

/* record the current CPU goverors to `state`
*/
Result get_state(State *state);

/* set the CPU governor to `state`
*/
Result set_state(const State &state);

/* save the current governor, to be used with restore()
*/
Result record();

/* restore the governor last captured with record()
*/
Result restore();

} // namespace turbo