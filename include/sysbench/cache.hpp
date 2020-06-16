#pragma once

#include <cstdlib>

void flush(void *p);

inline void barrier_all();

inline void flush_all(void *p, const size_t n);
