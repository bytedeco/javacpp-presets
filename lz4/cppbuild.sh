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
        ;;
    windows-x86)
        make -j $MAKEJ
        PREFIX=$INSTALL_PATH make install
        ;;
    windows-x86_64)
        make -j $MAKEJ
        PREFIX=$INSTALL_PATH make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
