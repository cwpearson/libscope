#pragma once

namespace turbo {


enum class Result {
  NO_PERMISSION,
  NOT_SUPPORTED,
  SUCCESS,
  UNKNOWN
};

const char *get_string(const Result &result);

struct State {
    State();
    bool valid;
    bool enabled;
};

bool can_modify();

Result enable();
Result disable();

Result get_state(State *state);
Result set_state(const State &state);

/* set from the global state*/
Result set_state();

/* record the global state*/
Result get_state();

} // namespace turbo