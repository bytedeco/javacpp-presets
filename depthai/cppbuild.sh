#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" depthai
    popd
    exit
fi

DEPTHAI_VERSION_COMMIT=b83eac2e5b623e357f1ef5a424f00fb1d5d0ac12
DEPTHAI_VERSION=2.13.3-b83eac2e5b623e357f1ef5a424f00fb1d5d0ac12
LIBUSB_VERSION=1.0.22

# download https://github.com/luxonis/depthai-core/releases/download/v$DEPTHAI_VERSION/depthai-core-v$DEPTHAI_VERSION.tar.gz depthai-core-v$DEPTHAI_VERSION.tar.gz
# download http://sourceforge.net/projects/libusb/files/libusb-1.0/libusb-$LIBUSB_VERSION/libusb-$LIBUSB_VERSION.tar.bz2/download libusb-$LIBUSB_VERSION.tar.bz2

# download https://github.com/luxonis/depthai-core/archive/b83eac2e5b623e357f1ef5a424f00fb1d5d0ac12.tar.gz depthai-core-$DEPTHAI_VERSION_COMMIT.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib bin
INSTALL_PATH=`pwd`

# echo "Decompressing archives..."
# tar --totals -xzf ../depthai-core-$DEPTHAI_VERSION_COMMIT.tar.gz

git clone https://github.com/luxonis/depthai-core.git depthai-core-$DEPTHAI_VERSION_COMMIT --recursive
cd depthai-core-$DEPTHAI_VERSION_COMMIT
git checkout $DEPTHAI_VERSION_COMMIT
git submodule update --init --recursive
# patch
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
    android-arm)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-armeabi-v7a/ -DANDROID_ABI=armeabi-v7a -DANDROID_PLATFORM=24 -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    android-arm64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-arm64-v8a/ -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=24 -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-x86/ -DANDROID_ABI=x86 -DANDROID_PLATFORM=24 -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    android-x86_64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-x86_64/ -DANDROID_ABI=x86_64 -DANDROID_PLATFORM=24 -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-armhf)
        echo 'set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")'   >> custom_toolchain.cmake
        echo 'set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")' >> custom_toolchain.cmake
        "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=$PWD/custom_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-arm64)
        echo 'set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")'   >> custom_toolchain.cmake
        echo 'set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")' >> custom_toolchain.cmake
        "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=$PWD/custom_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86)
        echo 'set(CMAKE_C_COMPILER "gcc")'            >> custom_toolchain.cmake
        echo 'set(CMAKE_C_FLAGS "-m32")'              >> custom_toolchain.cmake
        echo 'set(CMAKE_CXX_COMPILER "g++")'          >> custom_toolchain.cmake
        echo 'set(CMAKE_CXX_FLAGS "-m32")'            >> custom_toolchain.cmake
        "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=$PWD/custom_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        echo 'set(CMAKE_C_COMPILER "gcc")'            >> custom_toolchain.cmake
        echo 'set(CMAKE_CXX_COMPILER "g++")'          >> custom_toolchain.cmake
        "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=$PWD/custom_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DCMAKE_MACOSX_RPATH=ON -DBUILD_SHARED_LIBS=ON -DDEPTHAI_OPENCV_SUPPORT=ON .
        make -j $MAKEJ
        make install/strip
        install_name_tool -change /usr/local/opt/libusb/lib/libusb-1.0.0.dylib @rpath/libusb-1.0.0.dylib ../lib/libdepthai-core.dylib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
