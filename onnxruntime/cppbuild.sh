#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" onnxruntime
    popd
    exit
fi

export CUDACXX="/usr/local/cuda/bin/nvcc"
export CUDA_HOME="/usr/local/cuda"
export CUDNN_HOME="/usr/local/cuda"
export MAKEFLAGS="-j $MAKEJ"
export PYTHON_BIN_PATH=$(which python3)
if [[ $PLATFORM == windows* ]]; then
    export CUDACXX="$CUDA_PATH/bin/nvcc"
    export CUDA_HOME="$CUDA_PATH"
    export CUDNN_HOME="$CUDA_PATH"
    export PYTHON_BIN_PATH=$(which python.exe)
fi

export GPU_FLAGS=
if [[ "$EXTENSION" == *gpu ]]; then
    GPU_FLAGS="--use_cuda"
fi

ONNXRUNTIME=1.1.2

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`
mkdir -p build include lib bin

if [[ ! -d onnxruntime ]]; then
    git clone https://github.com/microsoft/onnxruntime
fi
cd onnxruntime
git reset --hard
git checkout v$ONNXRUNTIME
git submodule update --init --recursive --jobs $MAKEJ
git submodule foreach --recursive 'git reset --hard'

# work around toolchain issues on Mac and Windows
patch -p1 < ../../../onnxruntime.patch
sedinplace 's/CMAKE_ARGS/CMAKE_ARGS -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF/g' cmake/external/dnnl.cmake
sedinplace 's/HOST_NAME_MAX/sysconf(_SC_HOST_NAME_MAX)/g' onnxruntime/core/providers/cuda/cuda_call.cc

# use PTX instead of compiling for all CUDA archs to reduce library size
sedinplace 's/-gencode=arch=compute_30,code=sm_30/-arch=sm_30/g' cmake/CMakeLists.txt
sedinplace '/-gencode=arch=compute_..,code=sm_../d' cmake/CMakeLists.txt

which ctest3 &> /dev/null && CTEST="ctest3" || CTEST="ctest"
"$PYTHON_BIN_PATH" tools/ci_build/build.py --build_dir ../build --config Release --cmake_path "$CMAKE" --ctest_path "$CTEST" --build_shared_lib --use_dnnl --use_openmp $GPU_FLAGS

# install headers and libraries in standard directories
cp -r include/* ../include
cp ../build/Release/lib* ../lib || true
cp ../build/Release/Release/onnxruntime.dll ../bin || true
cp ../build/Release/Release/onnxruntime.lib ../lib || true

cd ../..
