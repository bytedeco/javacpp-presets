#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" onnxruntime
    popd
    exit
fi

export MAKEFLAGS="-j $MAKEJ"
export PYTHON_BIN_PATH=$(which python3)
if [[ $PLATFORM == windows* ]]; then
    export PYTHON_BIN_PATH=$(which python.exe)
fi

ONNXRUNTIME=1.1.1

mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`
mkdir -p build include lib bin

if [[ ! -d onnxruntime ]]; then
    git clone https://github.com/microsoft/onnxruntime
fi
cd onnxruntime
git reset --hard
git checkout v$ONNXRUNTIME
git submodule update --init --recursive --jobs $MAKEJ
git submodule foreach --recursive git reset --hard
patch -p1 < ../../../onnxruntime.patch
sedinplace 's/CMAKE_ARGS/CMAKE_ARGS -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF/g' cmake/external/dnnl.cmake
which ctest3 &> /dev/null && CTEST="ctest3" || CTEST="ctest"
"$PYTHON_BIN_PATH" tools/ci_build/build.py --build_dir ../build --config Release --cmake_path "$CMAKE" --ctest_path "$CTEST" --build_shared_lib --use_dnnl --use_openmp

cp -r include/* ../include
cp ../build/Release/lib* ../lib || true
cp ../build/Release/Release/onnxruntime.dll ../bin || true
cp ../build/Release/Release/onnxruntime.lib ../lib || true

cd ../..
