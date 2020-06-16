# 3.17 for FindCUDAToolkit
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(sysbench LANGUAGES CXX CUDA VERSION 0.1.0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

option(USE_NUMA ON)
option(USE_NVTX ON)
option(USE_OPENMP ON)
option(USE_CUDA ON)

# Get the git version
include(Git)
git_get_head_revision(SYSBENCH_GIT_REFSPEC SYSBENCH_GIT_HASH)
git_local_changes(SYSBENCH_GIT_LOCAL_CHANGES)
message(STATUS SYSBENCH_GIT_REFSPEC=${SYSBENCH_GIT_REFSPEC})
message(STATUS SYSBENCH_GIT_HASH=${SYSBENCH_GIT_HASH})
message(STATUS SYSBENCH_GIT_LOCAL_CHANGES=${SYSBENCH_GIT_LOCAL_CHANGES})

# Generate version file
message(STATUS "${PROJECT_SOURCE_DIR}/.../config.hpp.in -> ${PROJECT_BINARY_DIR}/.../config.hpp")
configure_file (
    "${PROJECT_SOURCE_DIR}/include/sysbench/config.hpp.in"
    "${PROJECT_BINARY_DIR}/include/sysbench/config.hpp"
)

# add Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "blah") # remove dep on gtest
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "blah") # remove dep on gtest
add_subdirectory(thirdparty/benchmark)
# google benchmark doesn't seem to export benchmark::benchmark target as promised
add_library(benchmark::benchmark ALIAS benchmark)

# Add spdlog
add_subdirectory(thirdparty/spdlog)
# google benchmark doesn't seem to export benchmark::benchmark target as promised
add_library(benchmark::benchmark ALIAS benchmark)

add_subdirectory(src)

add_library(sysbench
  ${SYSBENCH_CPP_SOURCES}
  ${SYSBENCH_CU_SOURCES}
)

target_include_directories(sysbench PUBLIC
  include
)

# Request that sysbench be built with -std=c++11
# As this is a public compile feature anything that links to
# sysbench will also build with -std=c++11
target_compile_features(sysbench PUBLIC cxx_std_11)

# Link against libraries
target_link_libraries(sysbench PUBLIC benchmark::benchmark)
target_link_libraries(sysbench PUBLIC spdlog::spdlog)

## CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
target_link_libraries(sysbench PUBLIC CUDA::cudart)
target_compile_definitions(sysbench PUBLIC -DSYSBENCH_USE_CUDA=1)

## NUMA
if (USE_NUMA)
  find_package(NUMA)
  if(NUMA_FOUND)
      message(STATUS "Found NUMA: " ${NUMA_INCLUDE_DIR}, ${NUMA_LIBRARY})
      target_compile_definitions(sysbench PUBLIC -DSYSBENCH_USE_NUMA=1)
      target_link_libraries(sysbench PUBLIC ${NUMA_LIBRARY})
      target_include_directories(sysbench SYSTEM PUBLIC ${NUMA_INCLUDE_DIR})
  else()
  target_compile_definitions(sysbench PUBLIC -DSYSBENCH_USE_NUMA=0)
    message(WARNING "didn't find NUMA, some benchmarks will be unavailable.")
  endif(NUMA_FOUND)
else(USE_NUMA)
endif(USE_NUMA)

## OpenMP
find_package(OpenMP)
if (OpenMP_FOUND)
  target_link_libraries(sysbench PUBLIC ${OpenMP_CXX_LIBRARIES})
  if (OpenMP_C_FLAGS)
    string(REPLACE ";" " " OpenMP_C_FLAGS ${OpenMP_C_FLAGS})
  endif()
  if (OpenMP_CXX_FLAGS)
    string(REPLACE ";" " " OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS})
  endif()

  ## C++ flags
  target_compile_options(sysbench PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
  )

  # CUDA flags
  foreach(flag IN ITEMS ${OpenMP_CXX_FLAGS})
    target_compile_options( sysbench PUBLIC
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag};>
    )
  endforeach(flag)

  add_definitions(-DUSE_OPENMP=1)
else(OpenMP_FOUND)
  add_definitions(-DUSE_OPENMP=0)
  message(WARNING "didn't find OpenMP, some benchmarks will be unavailable.")
endif(OpenMP_FOUND)

target_compile_features(sysbench PUBLIC cxx_std_11)
set_property(TARGET sysbench PROPERTY CUDA_STANDARD 11)


add_library(sysbench::sysbench ALIAS sysbench)

