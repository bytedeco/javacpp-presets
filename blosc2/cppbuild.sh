#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" blosc2
    popd
    exit
fi

BLOSC2_VERSION=3.6.2
download https://github.com/Blosc/c-blosc2/archive/refs/tags/v$BLOSC2_VERSION.tar.gz c-blosc2-$BLOSC2_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../c-blosc2-$BLOSC2_VERSION.tar.gz
cd c-blosc2-$BLOSC2_VERSION

# Build a single shared library that bundles its own LZ4, Zlib(-ng) and Zstd codecs,
# which is all that is needed to read and write Blosc1 and Blosc2 chunks, frames, and N-dimensional arrays
# (the optional plugins, tests, benchmarks, and examples are not needed)
CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_SHARED=ON -DBUILD_STATIC=OFF -DBUILD_TESTS=OFF -DBUILD_FUZZERS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PLUGINS=OFF"

case $PLATFORM in
    linux-arm64)
        export PREFIX=aarch64-linux-gnu
        CC=$PREFIX-gcc "$CMAKE" -DCMAKE_C_COMPILER=$PREFIX-gcc -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu $CMAKE_OPTIONS .
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        "$CMAKE" -DCMAKE_C_FLAGS="-m64 -fPIC" $CMAKE_OPTIONS .
        make -j $MAKEJ
        make install
        ;;
    macosx-arm64)
        "$CMAKE" -DCMAKE_OSX_ARCHITECTURES=arm64 $CMAKE_OPTIONS .
        make -j $MAKEJ
        make install
        ;;
    macosx-x86_64)
        "$CMAKE" -DCMAKE_OSX_ARCHITECTURES=x86_64 $CMAKE_OPTIONS .
        make -j $MAKEJ
        make install
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"
        "$CMAKE" -G "Ninja" $CMAKE_OPTIONS .
        ninja -j $MAKEJ
        ninja install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
