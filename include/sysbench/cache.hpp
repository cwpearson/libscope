#pragma once

#include <cstdlib>

void flush(void *p);

void barrier_all();

void flush_all(void *p, const size_t n);
