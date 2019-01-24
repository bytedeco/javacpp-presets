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

NCURSES=6.1
NGRAPH=0.11.1
download https://ftp.gnu.org/pub/gnu/ncurses/ncurses-$NCURSES.tar.gz ncurses-$NCURSES.tar.gz
download https://github.com/NervanaSystems/ngraph/archive/v$NGRAPH.tar.gz ngraph-$NGRAPH.tar.gz

mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xf ../ncurses-$NCURSES.tar.gz
tar --totals -xf ../ngraph-$NGRAPH.tar.gz

export LIBRARY_PATH="$INSTALL_PATH/lib"
export PATH="$PATH:$INSTALL_PATH/bin"
export CFLAGS="-I$INSTALL_PATH/include"
export CXXFLAGS="-I$INSTALL_PATH/include"

cd ncurses-$NCURSES
./configure "--prefix=$INSTALL_PATH" "--with-shared" CFLAGS=-fPIC CXXFLAGS=-fPIC
make -j $MAKEJ V=0
make install

ln -sf libncurses.so.6 ../lib/libtinfo.so.6
ln -sf libncurses.so.6 ../lib/libtinfo.so

cd ../ngraph-$NGRAPH
patch -Np1 < ../../../ngraph.patch

sedinplace '/In-source builds are not allowed/d' CMakeLists.txt
$CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR=lib -DNGRAPH_UNIT_TEST_ENABLE=FALSE -DNGRAPH_TOOLS_ENABLE=FALSE -DNGRAPH_ONNX_IMPORT_ENABLE=ON -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_ONNXIFI_ENABLE=TRUE
make -j $MAKEJ
make install

cp onnx/bin/libonnxifi.so ../lib/
cp onnx/src/onnx/onnxifi.h ../include/ngraph/frontend/onnxifi/
sedinplace  's/#include <onnxifi.h>/#include "onnxifi.h"/g' ../include/ngraph/frontend/onnxifi/backend_manager.hpp

cd ../..
