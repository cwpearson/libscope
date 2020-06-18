#pragma once

namespace turbo {

enum class Result { NO_PERMISSION, NOT_SUPPORTED, SUCCESS, UNKNOWN };

const char *get_string(const Result &result);

struct State {
  State();
  bool valid;
  bool enabled;
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