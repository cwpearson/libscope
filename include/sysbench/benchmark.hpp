#pragma once

// to be used in benchmark loop
#define OR_SKIP_AND_BREAK(stmt, msg)                                           \
  if (PRINT_IF_ERROR(stmt)) {                                                  \
    state.SkipWithError(msg);                                                  \
    break;                                                                     \
  }

// during setup or teardown
#define OR_SKIP(stmt, msg)                                                     \
  if (PRINT_IF_ERROR(stmt)) {                                                  \
    state.SkipWithError(msg);                                                  \
  }

#define OR_SKIP_AND_RETURN(stmt, msg)                                          \
  if (PRINT_IF_ERROR(stmt)) {                                                  \
    state.SkipWithError(msg);                                                  \
    return;                                                                    \
  }
