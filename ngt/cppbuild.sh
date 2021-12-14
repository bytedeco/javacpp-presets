#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" ngt
    popd
    exit
fi

# Must be kept in sync with ngt.version in pom.xml
NGT_VERSION=1.13.8
download https://github.com/yahoojapan/NGT/archive/refs/tags/v$NGT_VERSION.tar.gz ngt-$NGT_VERSION.tar.gz

mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xzf ../ngt-$NGT_VERSION.tar.gz

cd NGT-$NGT_VERSION
mkdir -p build
cd build

case $PLATFORM in
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -std=c++11 -m64"
        "$CMAKE" -DNGT_SHARED_MEMORY_ALLOCATOR=ON -DNGT_LARGE_DATASET=ON -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release ..
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
