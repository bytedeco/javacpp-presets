#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" sentencepiece
    popd
    exit
fi

SENTENCEPIECE_VERSION=0.1.99
download https://github.com/google/sentencepiece/archive/refs/tags/v$SENTENCEPIECE_VERSION.tar.gz sentencepiece-$SENTENCEPIECE_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar -xzvf ../sentencepiece-$SENTENCEPIECE_VERSION.tar.gz
cd sentencepiece-$SENTENCEPIECE_VERSION

CMAKE_CONFIG="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR=$INSTALL_PATH/lib"

mkdir -p build
cd build
$CMAKE $CMAKE_CONFIG ..
make -j $MAKEJ
make install

cd ../..
