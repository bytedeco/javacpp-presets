#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" librealsense2
    popd
    exit
fi

LIBREALSENSE2_VERSION=2.50.0
LIBUSB_VERSION=1.0.22
download https://github.com/IntelRealSense/librealsense/archive/v$LIBREALSENSE2_VERSION.tar.gz librealsense-$LIBREALSENSE2_VERSION.tar.gz
download http://sourceforge.net/projects/libusb/files/libusb-1.0/libusb-$LIBUSB_VERSION/libusb-$LIBUSB_VERSION.tar.bz2/download libusb-$LIBUSB_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib bin
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../librealsense-$LIBREALSENSE2_VERSION.tar.gz
tar --totals -xjf ../libusb-$LIBUSB_VERSION.tar.bz2

cd librealsense-$LIBREALSENSE2_VERSION
patch -Np1 < ../../../librealsense2.patch || true
sedinplace 's/float_t/float/g' `find src/tm2/ -type f`

case $PLATFORM in
    linux-armhf)
        cd ../libusb-$LIBUSB_VERSION
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf --disable-udev
        make -j $MAKEJ
        make install
        cd ../librealsense-$LIBREALSENSE2_VERSION
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLIBUSB_INC=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB_LIB=$INSTALL_PATH/lib/libusb-1.0.a -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-arm64)
        cd ../libusb-$LIBUSB_VERSION
        CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ CFLAGS="-march=armv8-a -mcpu=cortex-a57" CXXFLAGS="-march=armv8-a -mcpu=cortex-a57" CPPFLAGS="-march=armv8-a -mcpu=cortex-a57" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux-gnu --disable-udev
        make -j $MAKEJ
        make install
        cd ../librealsense-$LIBREALSENSE2_VERSION
        CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLIBUSB_INC=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB_LIB=$INSTALL_PATH/lib/libusb-1.0.a -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86)
        cd ../libusb-$LIBUSB_VERSION
        CC="gcc -m32" CXX="g++ -m32" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../librealsense-$LIBREALSENSE2_VERSION
        CC="gcc -m32" CXX="g++ -m32" "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLIBUSB_INC=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB_LIB=$INSTALL_PATH/lib/libusb-1.0.a -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        cd ../libusb-$LIBUSB_VERSION
        CC="gcc -m64" CXX="g++ -m64" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../librealsense-$LIBREALSENSE2_VERSION
        CC="gcc -m64" CXX="g++ -m64" "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLIBUSB_INC=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB_LIB=$INSTALL_PATH/lib/libusb-1.0.a -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_MACOSX_RPATH=ON -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF .
        make -j $MAKEJ
        make install/strip
        install_name_tool -change /usr/local/opt/libusb/lib/libusb-1.0.0.dylib @rpath/libusb-1.0.0.dylib ../lib/librealsense2.dylib
        ;;
    windows-x86)
        mkdir -p build
        cd build
        export CC="cl.exe"
        export CXX="cl.exe"
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF ..
        ninja -j $MAKEJ
        cd ..
        cp -a include/* ../include/
        cp -a build/* ../lib/
        ;;
    windows-x86_64)
        mkdir -p build
        cd build
        export CC="cl.exe"
        export CXX="cl.exe"
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF ..
        ninja -j $MAKEJ
        cd ..
        cp -a include/* ../include/
        cp -a build/* ../lib/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
