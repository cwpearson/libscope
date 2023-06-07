#pragma once

#include <unistd.h>

inline size_t page_size() { return sysconf(_SC_PAGESIZE); }