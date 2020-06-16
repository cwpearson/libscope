include(FindPackageHandleStandardArgs)

SET(NUMA_INCLUDE_SEARCH_PATHS
      ${NUMA}
      /usr/include
      $ENV{NUMA}
      $ENV{NUMA_HOME}
      $ENV{NUMA_HOME}/include
)

SET(NUMA_LIBRARY_SEARCH_PATHS
      ${NUMA}
      /usr/lib
      $ENV{NUMA}
      $ENV{NUMA_HOME}
      $ENV{NUMA_HOME}/lib
)

find_path(NUMA_INCLUDE_DIR
  NAMES numa.h
  PATHS ${NUMA_INCLUDE_SEARCH_PATHS}
  DOC "NUMA include directory")

find_library(NUMA_LIBRARY
  NAMES numa
  HINTS ${NUMA_LIBRARY_SEARCH_PATHS}
  DOC "NUMA library")

if (NUMA_LIBRARY)
    get_filename_component(NUMA_LIBRARY_DIR ${NUMA_LIBRARY} PATH)
endif()

mark_as_advanced(NUMA_INCLUDE_DIR NUMA_LIBRARY_DIR NUMA_LIBRARY)

find_package_handle_standard_args(NUMA REQUIRED_VARS NUMA_INCLUDE_DIR NUMA_LIBRARY)