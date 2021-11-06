#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" lz4
    popd
    exit
fi

LZ4_VERSION=1.9.3
download https://github.com/lz4/lz4/archive/refs/tags/v$LZ4_VERSION.tar.gz lz4-$LZ4_VERSION.tar.gz
mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
tar -xzvf ../lz4-$LZ4_VERSION.tar.gz
cd lz4-$LZ4_VERSION

# TODO add other platforms
case $PLATFORM in
    linux-x86)
        CC="gcc -m32 -fPIC" make -j $MAKEJ
        PREFIX=$INSTALL_PATH make install
        ;;
    linux-x86_64)
        CC="gcc -m64 -fPIC" make -j $MAKEJ
        PREFIX=$INSTALL_PATH make install
        ;;
    macosx-x86_64)
        make -j $MAKEJ
        PREFIX=$INSTALL_PATH make install
	# fix library with correct rpath
        install_name_tool -add_rpath @loader_path/. -id @rpath/liblz4.1.dylib ../lib/liblz4.1.dylib
        ;;
    windows-x86)
        cd build/cmake
        export CC="cl.exe"
        export CXX="cl.exe"
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_SHARED_LIBS=OFF .
        ninja -j $MAKEJ
        ninja install
        cd ../..
        ;;
    windows-x86_64)
        cd build/cmake
        export CC="cl.exe"
        export CXX="cl.exe"
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_SHARED_LIBS=OFF .
        ninja -j $MAKEJ
        ninja install
        cd ../..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
