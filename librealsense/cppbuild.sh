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

PATCH_ARCH=$(uname -m)

if [ "$PATCH_ARCH" == "loongarch64" ]; then
patch -Np1 -d libusb-$LIBUSB_VERSION < ../../libusb-add-loongarch-cpuinfo.patch
fi

cd librealsense-$LIBREALSENSE_VERSION
patch -Np1 --binary < ../../../librealsense.patch || true

case $PLATFORM in
    linux-armhf)
        cd ../libusb-$LIBUSB_VERSION
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf --disable-udev
        make -j $MAKEJ
        make install
        cd ../librealsense-$LIBREALSENSE_VERSION
        PKG_CONFIG_PATH="../lib/pkgconfig" CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ LDFLAGS="-lstdc++" "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLIBUSB1_INCLUDE_DIRS=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB1_LIBRARY_DIRS=$INSTALL_PATH/lib/ -DBUILD_UNIT_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-arm64)
        cd ../libusb-$LIBUSB_VERSION
        CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ CFLAGS="-march=armv8-a -mcpu=cortex-a57" CXXFLAGS="-march=armv8-a -mcpu=cortex-a57" CPPFLAGS="-march=armv8-a -mcpu=cortex-a57" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux-gnu --disable-udev
        make -j $MAKEJ
        make install
        cd ../librealsense-$LIBREALSENSE_VERSION
        PKG_CONFIG_PATH="../lib/pkgconfig" CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ LDFLAGS="-lstdc++" "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLIBUSB1_INCLUDE_DIRS=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB1_LIBRARY_DIRS=$INSTALL_PATH/lib/ -DBUILD_UNIT_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-loongarch64)
        cd ../libusb-$LIBUSB_VERSION
        CC="gcc -mabi=lp64" CXX="g++ -mabi=lp64" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=loongarch64-linux-gnu --disable-udev
        make -j $MAKEJ
        make install
        cd ../librealsense-$LIBREALSENSE_VERSION
        PKG_CONFIG_PATH="../lib/pkgconfig" CC="gcc -mabi=lp64" CXX="g++ -mabi=lp64 --std=c++11" LDFLAGS="-lstdc++" "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLIBUSB1_INCLUDE_DIRS=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB1_LIBRARY_DIRS=$INSTALL_PATH/lib/ -DBUILD_UNIT_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86)
        cd ../libusb-$LIBUSB_VERSION
        CC="gcc -m32" CXX="g++ -m32" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../librealsense-$LIBREALSENSE_VERSION
        PKG_CONFIG_PATH="../lib/pkgconfig" CC="gcc -m32" CXX="g++ -m32 --std=c++11" LDFLAGS="-lstdc++" "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLIBUSB1_INCLUDE_DIRS=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB1_LIBRARY_DIRS=$INSTALL_PATH/lib/ -DBUILD_UNIT_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        cd ../libusb-$LIBUSB_VERSION
        CC="gcc -m64" CXX="g++ -m64" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../librealsense-$LIBREALSENSE_VERSION
        PKG_CONFIG_PATH="../lib/pkgconfig" CC="gcc -m64" CXX="g++ -m64 --std=c++11" LDFLAGS="-lstdc++" "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLIBUSB1_INCLUDE_DIRS=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB1_LIBRARY_DIRS=$INSTALL_PATH/lib/ -DBUILD_UNIT_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        PKG_CONFIG_PATH="../lib/pkgconfig" "$CMAKE" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_MACOSX_RPATH=ON -DBUILD_UNIT_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        install_name_tool -change /usr/local/opt/libusb/lib/libusb-1.0.0.dylib @rpath/libusb-1.0.0.dylib ../lib/librealsense.dylib
        ;;
    windows-x86)
        export CC="cl.exe"
        export CXX="cl.exe"
        PKG_CONFIG_PATH="../lib/pkgconfig" "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_UNIT_TESTS=OFF .
        ninja -j $MAKEJ
        cp -a include/* ../include/
        cp -a *.lib *.dll ../lib/
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"
        PKG_CONFIG_PATH="../lib/pkgconfig" "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_UNIT_TESTS=OFF .
        ninja -j $MAKEJ
        cp -a include/* ../include/
        cp -a *.lib *.dll ../lib/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
