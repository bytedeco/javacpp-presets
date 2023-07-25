#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" sentencepiece
    popd
    exit
fi

SENTENCEPIECE_VERSION=0.1.99
download https://github.com/google/sentencepiece/archive/refs/tags/v$SENTENCEPIECE_VERSION.zip sentencepiece-$SENTENCEPIECE_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."

unzip -o ../sentencepiece-$SENTENCEPIECE_VERSION.zip

cd sentencepiece-$SENTENCEPIECE_VERSION

CMAKE_CONFIG="-DSPM_BUILD_TEST=ON -DSPM_ENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR=$INSTALL_PATH/lib"

mkdir -p build
cd build

case $PLATFORM in
    linux-arm64)
        export PREFIX=aarch64-linux-gnu
        CXX=$PREFIX-g++ CC=/$PREFIX-gcc $CMAKE $CMAKE_CONFIG -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu -DSPM_CROSS_SYSTEM_PROCESSOR=aarch64 ..
        ;;
    macosx-arm64)
        $CMAKE $CMAKE_CONFIG -DCMAKE_OSX_ARCHITECTURES="arm64" ..
        ;;
    *)
        $CMAKE $CMAKE_CONFIG ..
        ;;
esac

$CMAKE --build . --config Release --target install --parallel $MAKEJ

cd ../..
