set -x -e

source ci/env.sh

which cmake
which g++
if [[ $USE_CUDA != "0" ]]; then
  which nvcc
fi

cmake --version
g++ --version
if [[ $USE_CUDA != "0" ]]; then
  nvcc --version
fi

mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DSCOPE_USE_CUDA=$USE_CUDA \
  -DSCOPE_USE_NUMA=$USE_NUMA \
  -DSCOPE_USE_NVTX=$USE_NVTX \
  -DSCOPE_USE_OPENMP=$USE_OPENMP
make VERBOSE=1 