# 3.17 for FindCUDAToolkit
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(sysbench LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

option(SYSBENCH_USE_NUMA ON)
option(SYSBENCH_USE_NVTX ON)
option(SYSBENCH_USE_OPENMP ON)
option(SYSBENCH_USE_CUDA ON)

# Get the git version
include(Git)
git_get_head_revision(SYSBENCH_GIT_REFSPEC SYSBENCH_GIT_HASH)
git_local_changes(SYSBENCH_GIT_LOCAL_CHANGES)
message(STATUS SYSBENCH_GIT_REFSPEC=${SYSBENCH_GIT_REFSPEC})
message(STATUS SYSBENCH_GIT_HASH=${SYSBENCH_GIT_HASH})
message(STATUS SYSBENCH_GIT_LOCAL_CHANGES=${SYSBENCH_GIT_LOCAL_CHANGES})

# add Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "blah") # remove dep on gtest
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "blah") # remove dep on gtest
add_subdirectory(thirdparty/benchmark)
# google benchmark doesn't seem to export benchmark::benchmark target as promised
# add_library(benchmark::benchmark ALIAS benchmark)

# Add spdlog
add_subdirectory(thirdparty/spdlog)

# Add Lyra
add_subdirectory(thirdparty/Lyra)

add_subdirectory(src)

add_library(sysbench
  ${SYSBENCH_SOURCES}
)

target_include_directories(sysbench PUBLIC
  include
)

# Generate version file
message(STATUS "${PROJECT_SOURCE_DIR}/.../config.hpp.in -> ${PROJECT_BINARY_DIR}/.../config.hpp")
configure_file (
    "${PROJECT_SOURCE_DIR}/include/sysbench/config.hpp.in"
    "${PROJECT_BINARY_DIR}/include/sysbench/config.hpp"
)
target_include_directories(sysbench PUBLIC ${PROJECT_BINARY_DIR}/include/)

# enable CUDA if desired and available
include(CheckLanguage)
if(SYSBENCH_USE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(ERROR "No CUDA support")
  endif()
endif()

# check which languages are enabled
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

# Request that sysbench be built with -std=c++11
# As this is a public compile feature anything that links to
# sysbench will also build with -std=c++11
target_compile_features(sysbench PUBLIC cxx_std_11)
if("CUDA" IN_LIST languages)
  set_property(TARGET sysbench PROPERTY CUDA_STANDARD 11)
endif()

# Link against libraries
target_link_libraries(sysbench PUBLIC benchmark::benchmark)
target_link_libraries(sysbench PUBLIC spdlog::spdlog)
target_link_libraries(sysbench PUBLIC BFG::Lyra)

if("CUDA" IN_LIST languages)
    ## CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(sysbench PUBLIC CUDA::cudart)
    message(STATUS "CUDA found, compiling with -DSYSBENCH_USE_CUDA=1")
    target_compile_definitions(sysbench PUBLIC -DSYSBENCH_USE_CUDA=1)
else()
    target_compile_definitions(sysbench PUBLIC -DSYSBENCH_USE_CUDA=0)
endif()
    

## NUMA
if (SYSBENCH_USE_NUMA)
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
endif(SYSBENCH_USE_NUMA)

## OpenMP
if(SYSBENCH_USE_OPENMP)
  find_package(OpenMP REQUIRED)
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
endif(SYSBENCH_USE_OPENMP)

## NVTX
if(SYSBENCH_USE_NVTX AND "CUDA" IN_LIST languages)
  target_link_libraries(sysbench PUBLIC CUDA::nvToolsExt)
  target_compile_definitions(sysbench PUBLIC -DSYSBENCH_USE_NVTX=1)
else()
  target_compile_definitions(sysbench PUBLIC -DSYSBENCH_USE_NVTX=0)
endif()

add_library(sysbench::sysbench ALIAS sysbench)

