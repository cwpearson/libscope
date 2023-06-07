#pragma once

namespace turbo {

enum class Result { NO_PERMISSION, NOT_SUPPORTED, SUCCESS, UNKNOWN };
enum class Method {
  NONE,
  CPUFREQ,
  PSTATE,
};

const char *get_string(const Result &result);
const char *get_string(const Method &method);

struct State {
  State();
  bool enabled;
  Method method;
};

/* true if we are able to control the turbo state
 */
bool can_modify();

/* enable turbo
 */
Result enable();

/* disable turbo
 */
Result disable();

/* record current turbo state in `state`.
 */

Result get_state(State *state);
/* set turbo to `state`
 */
Result set_state(const State &state);

/* record the current turbo state into the global state
 */
Result get_state();

/* set turbo state from the global state
 */
Result set_state();

} // namespace turbo