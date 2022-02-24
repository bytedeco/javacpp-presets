#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" depthai
    popd
    exit
fi

DEPTHAI_VERSION=2.14.1
LIBUSB_VERSION=1.0.22
download https://github.com/luxonis/depthai-core/releases/download/v$DEPTHAI_VERSION/depthai-core-v$DEPTHAI_VERSION.tar.gz depthai-core-v$DEPTHAI_VERSION.tar.gz
download http://sourceforge.net/projects/libusb/files/libusb-1.0/libusb-$LIBUSB_VERSION/libusb-$LIBUSB_VERSION.tar.bz2/download libusb-$LIBUSB_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib bin
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../depthai-core-v$DEPTHAI_VERSION.tar.gz
tar --totals -xjf ../libusb-$LIBUSB_VERSION.tar.bz2

cd depthai-core-v$DEPTHAI_VERSION
#patch -Np1 < ../../../depthai.patch
sedinplace '/find_package(Git/d' CMakeLists.txt cmake/GitCommitHash.cmake shared/depthai-bootloader-shared.cmake shared/depthai-shared.cmake
sedinplace '/protected:/d' include/depthai/pipeline/Node.hpp

OPENCV_PATH=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -d "$P/include/opencv2" ]]; then
            OPENCV_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

export OpenCV_DIR="$OPENCV_PATH/lib/cmake/opencv4"

case $PLATFORM in
    linux-armhf)
        cd ../libusb-$LIBUSB_VERSION
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf --disable-udev
        make -j $MAKEJ
        make install
        cd ../depthai-core-v$DEPTHAI_VERSION
        echo 'set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")'   >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_C_FLAGS "-std=gnu99")'                   >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")' >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_CXX_FLAGS "-std=c++11")'                 >> cmake/toolchain/pic.cmake
        sedinplace "/    XLink/a CMAKE_ARGS LIBUSB_INCLUDE_DIR=$INSTALL_PATH/include/libusb-1.0/ LIBUSB_LIBRARY=$INSTALL_PATH/lib/libusb-1.0.a" cmake/Hunter/config.cmake
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DDEPTHAI_ENABLE_BACKWARD=OFF -DBUILD_SHARED_LIBS=ON -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-arm64)
        cd ../libusb-$LIBUSB_VERSION
        CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux-gnu --disable-udev
        make -j $MAKEJ
        make install
        cd ../depthai-core-v$DEPTHAI_VERSION
        echo 'set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")'   >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_C_FLAGS "-std=gnu99")'                 >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")' >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_CXX_FLAGS "-std=c++11")'               >> cmake/toolchain/pic.cmake
        sedinplace "/    XLink/a CMAKE_ARGS LIBUSB_INCLUDE_DIR=$INSTALL_PATH/include/libusb-1.0/ LIBUSB_LIBRARY=$INSTALL_PATH/lib/libusb-1.0.a" cmake/Hunter/config.cmake
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DDEPTHAI_ENABLE_BACKWARD=OFF -DBUILD_SHARED_LIBS=ON -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86)
        cd ../libusb-$LIBUSB_VERSION
        CC="gcc -m32" CXX="g++ -m32" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../depthai-core-v$DEPTHAI_VERSION
        echo 'set(CMAKE_C_COMPILER "gcc")'            >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_C_FLAGS "-m32 -std=gnu99")'   >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_CXX_COMPILER "g++")'          >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_CXX_FLAGS "-m32 -std=c++11")' >> cmake/toolchain/pic.cmake
        sedinplace "/    XLink/a CMAKE_ARGS LIBUSB_INCLUDE_DIR=$INSTALL_PATH/include/libusb-1.0/ LIBUSB_LIBRARY=$INSTALL_PATH/lib/libusb-1.0.a" cmake/Hunter/config.cmake
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DDEPTHAI_ENABLE_BACKWARD=OFF -DBUILD_SHARED_LIBS=ON -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        cd ../libusb-$LIBUSB_VERSION
        CC="gcc -m64" CXX="g++ -m64" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../depthai-core-v$DEPTHAI_VERSION
        echo 'set(CMAKE_C_COMPILER "gcc")'            >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_C_FLAGS "-m64 -std=gnu99")'   >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_CXX_COMPILER "g++")'          >> cmake/toolchain/pic.cmake
        echo 'set(CMAKE_CXX_FLAGS "-m64 -std=c++11")' >> cmake/toolchain/pic.cmake
        sedinplace "/    XLink/a CMAKE_ARGS LIBUSB_INCLUDE_DIR=$INSTALL_PATH/include/libusb-1.0/ LIBUSB_LIBRARY=$INSTALL_PATH/lib/libusb-1.0.a" cmake/Hunter/config.cmake
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DDEPTHAI_ENABLE_BACKWARD=OFF -DBUILD_SHARED_LIBS=ON -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DCMAKE_MACOSX_RPATH=ON -DDEPTHAI_ENABLE_BACKWARD=OFF -DBUILD_SHARED_LIBS=ON -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        install_name_tool -change /usr/local/opt/libusb/lib/libusb-1.0.0.dylib @rpath/libusb-1.0.0.dylib ../lib/libdepthai-core.dylib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
