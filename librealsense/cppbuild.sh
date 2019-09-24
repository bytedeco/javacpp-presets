#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" librealsense
    popd
    exit
fi

LIBREALSENSE_VERSION=1.12.4
LIBUSB_VERSION=1.0.22
download https://github.com/IntelRealSense/librealsense/archive/v$LIBREALSENSE_VERSION.tar.gz librealsense-$LIBREALSENSE_VERSION.tar.gz
download http://sourceforge.net/projects/libusb/files/libusb-1.0/libusb-$LIBUSB_VERSION/libusb-$LIBUSB_VERSION.tar.bz2/download libusb-$LIBUSB_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib bin
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../librealsense-$LIBREALSENSE_VERSION.tar.gz
tar --totals -xjf ../libusb-$LIBUSB_VERSION.tar.bz2

cd librealsense-$LIBREALSENSE_VERSION
patch -Np1 --binary < ../../../librealsense.patch || true

case $PLATFORM in
    linux-x86)
        cd ../libusb-$LIBUSB_VERSION
        CC="gcc -m32" CXX="g++ -m32" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../librealsense-$LIBREALSENSE_VERSION
        CC="gcc -m32" CXX="g++ -m32 --std=c++11" LDFLAGS="-lstdc++" "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLIBUSB1_INCLUDE_DIRS=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB1_LIBRARY_DIRS=$INSTALL_PATH/lib/ -DBUILD_UNIT_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        cd ../libusb-$LIBUSB_VERSION
        CC="gcc -m64" CXX="g++ -m64" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../librealsense-$LIBREALSENSE_VERSION
        CC="gcc -m64" CXX="g++ -m64 --std=c++11" LDFLAGS="-lstdc++" "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLIBUSB1_INCLUDE_DIRS=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB1_LIBRARY_DIRS=$INSTALL_PATH/lib/ -DBUILD_UNIT_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_MACOSX_RPATH=ON -DBUILD_UNIT_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86)
        "$CMAKE" -G "Visual Studio 15 2017" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_UNIT_TESTS=OFF .
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release
        cp -a include/* ../include/
        cp -a Release/* ../lib/
        ;;
    windows-x86_64)
        "$CMAKE" -G "Visual Studio 15 2017 Win64" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_UNIT_TESTS=OFF .
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release
        cp -a include/* ../include/
        cp -a Release/* ../lib/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
