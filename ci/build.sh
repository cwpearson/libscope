set -x -e

source ci/env.sh

which g++
which nvcc
which cmake

cmake --version
g++ --version
if [[ $USE_CUDA != "0" ]]; then
  nvcc --version
fi

mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DSYSBENCH_USE_CUDA=$USE_CUDA \
  -DSYSBENCH_USE_NUMA=$USE_NUMA \
  -DSYSBENCH_USE_NVTX=$USE_NVTX \
  -DSYSBENCH_USE_OPENMP=$USE_OPENMP
make VERBOSE=1 