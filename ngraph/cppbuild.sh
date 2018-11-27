#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" onnx
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    #No Windows support yet
    echo "Error: Platform \"$PLATFORM\" is not supported"
    exit 1
fi

export NGRAPH="0.10.0-rc.0"
export NCURSES=6.1

download https://github.com/NervanaSystems/ngraph/archive/v$NGRAPH.tar.gz ngraph.tar.gz
download https://ftp.gnu.org/pub/gnu/ncurses/ncurses-$NCURSES.tar.gz ncurses.tar.gz
mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xf ../ngraph.tar.gz
tar --totals -xf ../ncurses.tar.gz

cd ncurses-$NCURSES
./configure "--prefix=$INSTALL_PATH" CFLAGS=-fPIC CXXFLAGS=-fPIC
make -j $MAKEJ V=0
make install

cd ..

cd ngraph-$NGRAPH
rm -rf build
mkdir build && cd build
cmake .. -DNGRAPH_ONNX_IMPORT_ENABLE=ON -DCMAKE_INSTALL_PREFIX=~/ngraph_dist -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_ONNXIFI_ENABLE=TRUE
make -j 8
make install

mkdir -p ../../include/ngraph ../../include/ngraph/op ../../include/ngraph/pass ../../include/ngraph/state ../../include/ngraph/codegen ../../include/ngraph/op/util ../../include/ngraph/autodiff/ ../../include/ngraph/descriptor ../../include/ngraph/descriptor/layout ../../include/ngraph/runtime ../../include/ngraph/runtime/cpu ../../include/ngraph/type ../../include/onnx ../../lib

patch ../src/ngraph/frontend/onnxifi/backend.hpp ../../../../backend.hpp.patch
patch ../src/ngraph/frontend/onnxifi/backend_manager.hpp ../../../../backend_manager.hpp.patch
#patch ../src/ngraph/descriptor/tensor.hpp ../../../../tensor.hpp.patch
patch ../src/ngraph/type/element_type.hpp ../../../../element_type.hpp.patch

cp src/ngraph/frontend/onnxifi/libonnxifi-ngraph.so ../../lib/
cp src/ngraph/libngraph.so ../../lib/
cp onnx/bin/libonnxifi.so ../../lib/
cp onnx/src/onnx/onnxifi.h ../../include/ngraph/
cp onnx/src/onnx/onnxifi.h ../../include/onnx/
cp ../src/ngraph/frontend/onnx_import/onnx.hpp ../../include/ngraph/
cp ../src/ngraph/frontend/onnxifi/backend.hpp ../../include/ngraph/
cp ../src/ngraph/frontend/onnxifi/backend_manager.hpp ../../include/ngraph/
cp ../src/ngraph/runtime/backend.hpp ../../include/ngraph/runtime/backend.hpp
cp ../src/ngraph/runtime/tensor.hpp ../../include/ngraph/runtime/tensor.hpp
cp ../src/ngraph/runtime/performance_counter.hpp ../../include/ngraph/runtime/
cp ../src/ngraph/runtime/cpu/*.hpp ../../include/ngraph/runtime/cpu/
cp ../src/ngraph/type/*.hpp ../../include/ngraph/type/
cp ../src/ngraph/op/*.hpp ../../include/ngraph/op/
cp ../src/ngraph/op/util/*.hpp ../../include/ngraph/op/util/
cp ../src/ngraph/descriptor/*.hpp ../../include/ngraph/descriptor/
cp ../src/ngraph/descriptor/layout/*.hpp ../../include/ngraph/descriptor/layout/
cp ../src/ngraph/autodiff/*.hpp ../../include/ngraph/autodiff/
cp ../src/ngraph/codegen/*.hpp ../../include/ngraph/codegen/
cp ../src/ngraph/pass/*.hpp ../../include/ngraph/pass/
cp ../src/ngraph/state/*.hpp ../../include/ngraph/state/
cp ../src/ngraph/*.hpp ../../include/ngraph/

cd ../../..
