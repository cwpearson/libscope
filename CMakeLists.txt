# 3.17 for FindCUDAToolkit
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(SCOPE LANGUAGES CXX VERSION 1.1.2)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

option(SCOPE_USE_CUDA OFF)
option(SCOPE_USE_HIP OFF)
option(SCOPE_USE_NUMA ON)
option(SCOPE_USE_NVTX ON)
option(SCOPE_USE_OPENMP ON)

message(STATUS "SCOPE_USE_HIP:  ${SCOPE_USE_HIP}")
message(STATUS "SCOPE_USE_CUDA: ${SCOPE_USE_CUDA}")
message(STATUS "SCOPE_USE_NUMA: ${SCOPE_USE_NUMA}")
message(STATUS "SCOPE_USE_NVTX: ${SCOPE_USE_NVTX}")

# link libraries but make the includes system includes to suppress warnings
# https://stackoverflow.com/questions/52135983/cmake-target-link-libraries-include-as-system-to-suppress-compiler-warnings
function(target_link_libraries_system target)
  set(libs ${ARGN})
  foreach(lib ${libs})
    get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${target} SYSTEM PRIVATE ${lib_include_dirs})
    target_link_libraries(${target} PUBLIC ${lib})
  endforeach(lib)
endfunction(target_link_libraries_system)

# Get the git version
include(Git)
git_get_head_revision(SCOPE_GIT_REFSPEC SCOPE_GIT_HASH)
git_local_changes(SCOPE_GIT_LOCAL_CHANGES)
message(STATUS SCOPE_GIT_REFSPEC=${SCOPE_GIT_REFSPEC})
message(STATUS SCOPE_GIT_HASH=${SCOPE_GIT_HASH})
message(STATUS SCOPE_GIT_LOCAL_CHANGES=${SCOPE_GIT_LOCAL_CHANGES})

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

add_library(scope
  ${SCOPE_SOURCES}
)

add_subdirectory(bin)

# if we are included in another project with add_subdirectory(), mark our includes as system includes
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  target_include_directories(scope PUBLIC
    include
  )
else()
  target_include_directories(scope SYSTEM PUBLIC
    include
  )
endif()

# CXX flags
target_compile_options(scope PRIVATE
$<$<COMPILE_LANGUAGE:CXX>:
    -Wall; 
    -Wextra;
    -Wcast-qual;
    -Wcast-align;
    -Wstrict-aliasing;
    -Wpointer-arith;
    -Winit-self;
    -Wshadow;
    -Wswitch-enum;
    -Wredundant-decls;
    -Wfloat-equal;
    -Wundef;
    -Wvla;
    -fmax-errors=1;
    -Wfatal-errors;
>
  )

# CUDA flags
target_compile_options(scope PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo;
    -Xcompiler=-Wall; 
    -Xcompiler=-Wextra;
    -Xcompiler=-Wcast-qual;
    -Xcompiler=-Wcast-align;
    -Xcompiler=-Wstrict-aliasing;
    -Xcompiler=-Wpointer-arith;
    -Xcompiler=-Winit-self;
    -Xcompiler=-Wshadow;
    -Xcompiler=-Wswitch-enum;
    -Xcompiler=-Wfloat-equal;
    -Xcompiler=-Wvla;
    -Xcompiler=-fmax-errors=1;
    -Xcompiler=-Wfatal-errors;
>
)
# disable GCC notes
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
target_compile_options(scope PUBLIC
$<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-fcompare-debug-second 
>
  )
target_compile_options(scope PUBLIC
$<$<COMPILE_LANGUAGE:CXX>:
    -fcompare-debug-second 
>
  )
endif()

# Generate version file
message(STATUS "${PROJECT_SOURCE_DIR}/.../config.hpp.in -> ${PROJECT_BINARY_DIR}/.../config.hpp")
configure_file (
    "${PROJECT_SOURCE_DIR}/include/scope/config.hpp.in"
    "${PROJECT_BINARY_DIR}/include/scope/config.hpp"
)
target_include_directories(scope PUBLIC ${PROJECT_BINARY_DIR}/include/)

# enable CUDA if desired and available
include(CheckLanguage)
if(SCOPE_USE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(ERROR "No CUDA support")
  endif()
endif()

# check which languages are enabled
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

# Request that scope be built with -std=c++11
# As this is a public compile feature anything that links to
# scope will also build with -std=c++11
target_compile_features(scope PUBLIC cxx_std_11)
if("CUDA" IN_LIST languages)
  set_property(TARGET scope PROPERTY CUDA_STANDARD 11)
  set_property(TARGET scope PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Link against libraries
target_link_libraries_system(scope benchmark::benchmark)
target_link_libraries_system(scope spdlog::spdlog)
target_link_libraries_system(scope BFG::Lyra)

if("CUDA" IN_LIST languages)
    ## CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(scope PUBLIC CUDA::cudart)
    message(STATUS "CUDA found, compiling with -DSCOPE_USE_CUDA=1")
    target_compile_definitions(scope PUBLIC -DSCOPE_USE_CUDA=1)
else()
endif()

if(SCOPE_USE_HIP)
  target_compile_definitions(scope PUBLIC -DSCOPE_USE_HIP=1)
endif()

## NUMA
if (SCOPE_USE_NUMA)
  find_package(NUMA)
  if(NUMA_FOUND)
      message(STATUS "Found NUMA: " ${NUMA_INCLUDE_DIR}, ${NUMA_LIBRARY})
      target_compile_definitions(scope PUBLIC -DSCOPE_USE_NUMA=1)
      target_link_libraries(scope PUBLIC ${NUMA_LIBRARY})
      target_include_directories(scope SYSTEM PUBLIC ${NUMA_INCLUDE_DIR})
  else()
    message(WARNING "didn't find NUMA, some benchmarks will be unavailable.")
  endif(NUMA_FOUND)
else()
endif(SCOPE_USE_NUMA)

## OpenMP
if(SCOPE_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if (OpenMP_FOUND)
    target_link_libraries(scope PUBLIC ${OpenMP_CXX_LIBRARIES})
    if (OpenMP_C_FLAGS)
      string(REPLACE ";" " " OpenMP_C_FLAGS ${OpenMP_C_FLAGS})
    endif()
    if (OpenMP_CXX_FLAGS)
      string(REPLACE ";" " " OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS})
    endif()

    ## C++ flags
    target_compile_options(scope PUBLIC
      $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
    )

    # CUDA flags
    foreach(flag IN ITEMS ${OpenMP_CXX_FLAGS})
      target_compile_options( scope PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag};>
      )
    endforeach(flag)

    add_definitions(-DUSE_OPENMP=1)
  else(OpenMP_FOUND)
    message(WARNING "didn't find OpenMP, some benchmarks will be unavailable.")
  endif(OpenMP_FOUND)
endif(SCOPE_USE_OPENMP)

## NVTX
if(SCOPE_USE_NVTX AND "CUDA" IN_LIST languages)
  target_link_libraries(scope PUBLIC CUDA::nvToolsExt)
  target_compile_definitions(scope PUBLIC -DSCOPE_USE_NVTX=1)
else()
endif()

add_library(scope::scope ALIAS scope)

