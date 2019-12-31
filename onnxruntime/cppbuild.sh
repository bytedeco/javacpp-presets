#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" onnxruntime
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    #No Windows support yet
    echo "Error: Platform \"$PLATFORM\" is not supported"
    exit 1
fi

ONNXRUNTIME=1.1.0

mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`
mkdir -p include lib bin

if [[ ! -d onnxruntime ]]; then
    git clone https://github.com/microsoft/onnxruntime
fi
cd onnxruntime
git reset --hard
git checkout v$ONNXRUNTIME
git submodule update --init --recursive --jobs $MAKEJ
git submodule foreach --recursive git reset --hard
patch -p1 < ../../../onnxruntime.patch
which ctest3 &> /dev/null && CTEST="ctest3" || CTEST="ctest"
MAKEFLAGS="-j $MAKEJ" bash build.sh --cmake_path "$CMAKE" --ctest_path "$CTEST" --config Release --use_dnnl --use_mklml --build_shared_lib

cp -r include/* ../include
cp -r build/Linux/Release/lib* build/Linux/Release/dnnl/install/lib*/libdnnl* ../lib

cd ../..
