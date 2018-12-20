#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" ngraph
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    #No Windows support yet
    echo "Error: Platform \"$PLATFORM\" is not supported"
    exit 1
fi

export NGRAPH="0.10.1"
export NCURSES=6.1

download https://github.com/NervanaSystems/ngraph/archive/v$NGRAPH.tar.gz v$NGRAPH.tar.gz
download https://ftp.gnu.org/pub/gnu/ncurses/ncurses-$NCURSES.tar.gz ncurses.tar.gz
mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xf ../v$NGRAPH.tar.gz
tar --totals -xf ../ncurses.tar.gz

export LIBRARY_PATH="$INSTALL_PATH/lib"
export PATH="$PATH:$INSTALL_PATH/bin"
export CFLAGS="-I$INSTALL_PATH/include"
export CXXFLAGS="-I$INSTALL_PATH/include"

cd ncurses-$NCURSES
./configure "--prefix=$INSTALL_PATH" "--with-shared" CFLAGS=-fPIC CXXFLAGS=-fPIC
make -j $MAKEJ V=0
make install

cd ..

cd lib

ln -s libncurses.so.6 libtinfo.so.6
ln -s libncurses.so.6 libtinfo.so

cd ..

cd ngraph-$NGRAPH
rm -rf build
mkdir build && cd build
#-DNGRAPH_TBB_ENABLE=FALSE

#patch ../src/ngraph/runtime/cpu/CMakeLists.txt ../../../../CMakeLists.txt.patch
$CMAKE .. -DNGRAPH_UNIT_TEST_ENABLE=FALSE -DNGRAPH_TOOLS_ENABLE=FALSE -DNGRAPH_ONNX_IMPORT_ENABLE=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_ONNXIFI_ENABLE=TRUE
make -j $MAKEJ
make install

#mkdir -p ../../include/ngraph ../../include/ngraph/op ../../include/ngraph/pass ../../include/ngraph/state ../../include/ngraph/codegen ../../include/ngraph/op/util ../../include/ngraph/autodiff/ ../../include/ngraph/descriptor ../../include/ngraph/descriptor/layout ../../include/ngraph/runtime ../../include/ngraph/runtime/cpu ../../include/ngraph/type ../../include/onnx ../../lib

#mkdir ../../include/onnx

cp -r ../src/ngraph/frontend/onnx_import/core/ ../../include/
cp -r ../src/ngraph/frontend/onnx_import/utils/ ../../include/
cp -r ../src/ngraph/frontend/onnx_import/utils/ ../../include/ngraph/frontend/onnx_import/op/
cp -r ../src/ngraph/frontend/onnx_import/exceptions.hpp ../../include/


patch ../../include/ngraph/frontend/onnxifi/backend.hpp ../../../../backend.hpp.patch
patch ../../include/ngraph/frontend/onnxifi/backend_manager.hpp ../../../../backend_manager.hpp.patch
#patch ../src/ngraph/descriptor/tensor.hpp ../../../../tensor.hpp.patch


#execstack -c ../../lib/libtbb.so.2

cp src/ngraph/frontend/onnxifi/libonnxifi-ngraph.so ../../lib/
cp src/ngraph/libngraph.so ../../lib/
cp src/ngraph/libcpu_backend.so ../../lib/
cp src/ngraph/codegen/libcodegen.so ../../lib/
cp src/ngraph/runtime/cpu/tbb_build/tbb_release/libtbb.so.2 ../../lib/
cp onnx/bin/libonnxifi.so ../../lib/

ln -sf libtbb.so.2 ../../lib/libtbb.so

cp onnx/src/onnx/onnxifi.h ../../include/ngraph/frontend/onnxifi/
#cp onnx/src/onnx/onnxifi.h ../../include/onnx/
#cp ../src/ngraph/frontend/onnx_import/onnx.hpp ../../include/ngraph/
#cp ../src/ngraph/frontend/onnxifi/backend.hpp ../../include/ngraph/frontend/onnxifi/
#cp ../src/ngraph/frontend/onnxifi/backend_manager.hpp ../../include/ngraph/frontend/onnxifi/

cd ../../..
