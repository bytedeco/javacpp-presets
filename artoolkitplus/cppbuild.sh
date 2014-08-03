#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" artoolkitplus
    popd
    exit
fi

ARTOOLKITPLUS_VERSION=2.3.0
download https://launchpad.net/artoolkitplus/trunk/$ARTOOLKITPLUS_VERSION/+download/ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2 ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
tar -xjvf ../ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2 --exclude="*/id-markers/*"
cd ARToolKitPlus-$ARTOOLKITPLUS_VERSION
patch -Np1 < ../../../ARToolKitPlus-$ARTOOLKITPLUS_VERSION.patch

case $PLATFORM in
    android-arm)
        cmake -DCMAKE_TOOLCHAIN_FILE=android-arm.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
     android-x86)
        cmake -DCMAKE_TOOLCHAIN_FILE=android-x86.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    macosx-*)
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    windows-x86)
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        nmake
        nmake install
        ;;
    windows-x86_64)
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        nmake
        nmake install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
