#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" librealsense
    popd
    exit
fi

LIBREALSENSE_VERSION=1.12.1
download https://github.com/IntelRealSense/librealsense/archive/v$LIBREALSENSE_VERSION.tar.gz librealsense-$LIBREALSENSE_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib bin
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../librealsense-$LIBREALSENSE_VERSION.tar.gz

cd librealsense-$LIBREALSENSE_VERSION
patch -Np1 < ../../../librealsense.patch

case $PLATFORM in
    linux-x86)
        CC="gcc -m32" CXX="gcc -m32 --std=c++11" LDFLAGS="-lstdc++" "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH"
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="gcc -m64 --std=c++11" LDFLAGS="-lstdc++" "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH"
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_MACOSX_RPATH=ON
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86)
        "$CMAKE" -G "Visual Studio 14 2015" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH"
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release
        cp -a include/* ../include/
        cp -a Release/* ../lib/
        ;;
    windows-x86_64)
        "$CMAKE" -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH"
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release
        cp -a include/* ../include/
        cp -a Release/* ../lib/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
