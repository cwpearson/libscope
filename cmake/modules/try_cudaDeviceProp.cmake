

try_compile(
  SCOPE_HAVE_CUDA_DEVICE_PROP 
  ${CMAKE_BINARY_DIR} 
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/test_cudaDeviceProp.cpp
  OUTPUT_VARIABLE OUTPUT
)
message(STATUS "Performing Test SCOPE_HAVE_CUDA_DEVICE_PROP=${SCOPE_HAVE_CUDA_DEVICE_PROP}")