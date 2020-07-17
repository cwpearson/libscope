# libscope

A systems-oriented C++11 benchmark support library brining the following tools under one roof:
* CUDA support (including nvToolsExt)
* NUMA support
* OpenMP support
* CPU Cache Control (amd64 and ppc64le)
* CPU turbo control (linux)
* CPU governor control (amd64/linux and ppc64le/linux)
* [Google's Benchmark library](https://github.com/google/benchmark)
* [Lyra CLI parsing library](https://github.com/bfgroup/Lyra)
* [spdlog logging library](https://github.com/gabime/spdlog)

This work was started at the University of Illinois with Professor Wen-Mei Hwu's IMPACT research group in collaboration with IBM's T. J. Watson Research as the SCOPE project.
This project reworks the SCOPE framework as a library.

The [Comm|Scope](github.com/c3sr/comm_scope) multi-GPU communication benchmarking tool uses this library.

## Quickstart

Get CMake 3.17+ (needed for FindCUDAToolkit)

Add to your `CMakeLists.txt`:
```cmake
add_subdirectory(thirdparty/scope)
target_link_libraries(<target> scope::scope)
```

Include "scope/scope.hpp"
```c++
#include "scope/scope.hpp"

int main(int argc, char **argv) {
  // initialize scope framework things
  scope::init(&argc, argv);
  // run all registered benchmarks
  scope::run();
  // clean up scope
  scope::finalize();
}
```

Define a benchmark using [google/benchmark](https://github.com/google/benchmark).
Scope includes it built in and supports all google benchmark command line flags.

## How To

### Command Line Flags

All Scope applications support the following command line options:

* `--cuda <device ID>`: add GPU visibility (default: all). May be repeated to add more GPUs.
* `--numa <node ID>`: add NUMA visibility (default: all). May be repeated to add more NUMA nodes.

### CPU turbo (`scope/turbo.hpp`)

`scope::init()` will record the CPU's current turbo state, and attempt to disable it, if it is executed with sufficient permissions (sudo).
When `scope` exits from SIGINT or `finalize()`s, the original state will be restored.
Otherwise, use `enable-turbo` to enable CPU turbo again.

You may also programatically control the CPU turbo state with the following library functions:
```c++
namespace turbo {
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
}
```

### CPU governor (`scope/governor.hpp`)

`scope::init()` will record the current CPU governor, and attempt to set it to maximum it, if it is executed with sufficient permissions (sudo).
When `scope` exits from SIGINT or `finalize()`s, the original governor will be restored.
Otherwise, use `set-minimum` to restore the `powersave` governor.

You may also programatically control the CPU turbo state with the following library functions:
```c++
namespace governor {

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
```

### NUMA (`scope/numa.hpp`)

by default `scope` is compiled with NUMA support (SCOPE_USE_NUMA=1). It can be turned off with `cmake -DUSE_NUMA=0`.

Either way, the following API is exposed in the `numa` namespace.
If NUMA support is disabled, the API is consistent with a system that has a single NUMA domain with ID 0.

```c++
/* True if there is NUMA support and the system supports NUMA, false otherwise
 */
bool numa::available();

/* bind future processing and allocation by this thread to `node`.
If no NUMA support, does nothing
*/
void numa::bind_node(int node);

/* return the number of numa nodes
If no NUMA support, return 1
*/
int numa::node_count();

/* return the NUMA ids present in the system
 */
std::vector<int> numa::ids();
```

There is also a `numa::ScopedBind` class that is an RAII-wrapper around `numa::bind_node()`

```c++
// Code out here runs anywhere
{
numa::ScopedBind binder(13);
// this code now runs on node 13
}
// Code out here runs anywhere
```

### Cache Control (`scope/cache.hpp`)

```c++
// flush the cache line containing p
void flush(void *p);

// mfence (amd64) or sync 0 (ppc64le)
void barrier_all();

// flush all cache lines for the n-byte region starting at p
void flush_all(void *p, const size_t n);

```

## Roadmap

## Changelog

* v1.1.2 (July 17, 2020)
  * fix a bug in getting available NUMA nodes
* v1.1.1 (July 17, 2020)
  * fix a bug in getting available CUDA devices
* v1.1.0 (July 17, 2020)
  * Re-raise INT, HUP, and KILL signals after cleanup
  * add `--cuda` and `--numa` flags
  * Cache NUMA configuration to improve benchmark registration performance
* v1.0.0
  * Initial port from `c3sr/scope`
  * CPU governor API
  * CPU turbo API
  * [google/benchmark](https://github.com/google/benchmark) 1.5.1
  * [bfgroup/lyra](https://github.com/bfgroup/Lyra) 1.4.1
  * [gabime/spdlog](https://github.com/gabime/spdlog) 1.6.1

## Publications

```bibtex
@inproceedings{10.1145/3297663.3310299,
author = {Pearson, Carl and Dakkak, Abdul and Hashash, Sarah and Li, Cheng and Chung, I-Hsin and Xiong, Jinjun and Hwu, Wen-Mei},
title = {Evaluating Characteristics of CUDA Communication Primitives on High-Bandwidth Interconnects},
year = {2019},
isbn = {9781450362399},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3297663.3310299},
doi = {10.1145/3297663.3310299},
booktitle = {Proceedings of the 2019 ACM/SPEC International Conference on Performance Engineering},
pages = {209–218},
numpages = {10},
keywords = {nvlink, numa, power, x86, benchmarking, cuda, gpu},
location = {Mumbai, India},
series = {ICPE ’19}
}
```

```
@article{DBLP:journals/corr/abs-1809-08311,
  author    = {Carl Pearson and
               Abdul Dakkak and
               Cheng Li and
               Sarah Hashash and
               Jinjun Xiong and
               Wen{-}Mei W. Hwu},
  title     = {{SCOPE:} {C3SR} Systems Characterization and Benchmarking Framework},
  journal   = {CoRR},
  volume    = {abs/1809.08311},
  year      = {2018},
  url       = {http://arxiv.org/abs/1809.08311},
  archivePrefix = {arXiv},
  eprint    = {1809.08311},
  timestamp = {Fri, 05 Oct 2018 11:34:52 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1809-08311.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```
@inproceedings{pearson2018numa,
  title={NUMA-aware data-transfer measurements for power/NVLink multi-GPU systems},
  author={Pearson, Carl and Chung, I-Hsin and Sura, Zehra and Hwu, Wen-Mei and Xiong, Jinjun},
  booktitle={International Conference on High Performance Computing},
  pages={448--454},
  year={2018},
  organization={Springer}
}
```

## Acks
Thanks to Sarah Hashash (MIT), I-Hsin Chung (IBM T. J. Watson), and Jinjun Xiong (IBM T. J. Watson) for their support, guidance, and contributions.

Built with ❤️ using
  * [google/benchmark](https://github.com/google/benchmark)
  * [bfgroup/lyra](https://github.com/bfgroup/Lyra)
  * [gabime/spdlog](https://github.com/gabime/spdlog)

