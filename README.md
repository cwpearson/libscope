# sysbench

A systems-oriented benchmark support library bringing NUMA, CUDA, and amd64/ppc64le cache control under one roof.

* write benchmarks with programmatic NUMA pinning
* write benchmarks using CUDA
* flush CPU caches programmatically

This work was started at the University of Illinois with Professor Wen-Mei Hwu's IMPACT research group in collaboration with IBM's T. J. Watson Research.

The [Comm|Scope](github.com/c3sr/comm_scope) multi-GPU communication benchmarking tool uses this library.

## How To

### NUMA (`sysbench/numa.hpp`)

by default `sysbench` is compiled with NUMA support (USE_NUMA=1). It can be turned off with `cmake -DUSE_NUMA=0`.

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

### Cache Control (`sysbench/cache.hpp`)

```c++
// flush the cache line containing p
void flush(void *p);

// mfence (amd64) or sync 0 (ppc64le)
void barrier_all();

// flush all cache lines for the n-byte region starting at p
void flush_all(void *p, const size_t n);

```

## Roadmap
- [ ] Resolve name collision with [akopytov/sysbench](https://github.com/akopytov/sysbench)
  - [ ] SOBS (**S**ystem-**O**riented **B**enchmark **S**upport)
  - [ ] SOMBER (**S**ystem-**O**riented **M**icro**BE**nchma**R**k)
- [ ] Linux Performance Governor

## Changelog

* v0.1.0
  * Initial port from `c3sr/scope`
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

