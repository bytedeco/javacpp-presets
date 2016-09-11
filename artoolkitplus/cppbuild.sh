#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" artoolkitplus
    popd
    exit
fi

ARTOOLKITPLUS_VERSION=2.3.1
download https://launchpad.net/artoolkitplus/trunk/$ARTOOLKITPLUS_VERSION/+download/ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2 ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
tar -xjvf ../ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2 --exclude="*/id-markers/*"
cd ARToolKitPlus-$ARTOOLKITPLUS_VERSION
patch --binary -Np1 < ../../../ARToolKitPlus-$ARTOOLKITPLUS_VERSION.patch || true

case $PLATFORM in
    android-arm)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=android-arm.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
     android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=android-x86.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    linux-x86)
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    linux-x86_64)
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    linux-armhf)
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    linux-ppc64le)
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    macosx-*)
        $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    windows-x86)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        nmake
        nmake install
        ;;
    windows-x86_64)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        nmake
        nmake install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
