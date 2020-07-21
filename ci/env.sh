CMAKE_PREFIX=$HOME/cmake

# default enable all components
if [ -z ${USE_CUDA+x} ]; then USE_CUDA=1; fi
if [ -z ${USE_OPENMP+x} ]; then USE_OPENMP=1; fi
if [ -z ${USE_NUMA+x} ]; then USE_NUMA=1; fi
if [ -z ${USE_NVTX+x} ]; then USE_NVTX=1; fi

#default CUDA 11.0
if [ -z ${CUDA_VERSION+x} ]; then CUDA_VERSION=110; fi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export PATH=$CMAKE_PREFIX/bin:$PATH