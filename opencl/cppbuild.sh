#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" opencl
    popd
    exit
fi

OPENCL_VERSION=2023.12.14
CLHPP_VERSION=2023.12.14
download https://github.com/KhronosGroup/OpenCL-Headers/archive/v$OPENCL_VERSION.tar.gz OpenCL-Headers-$OPENCL_VERSION.tar.gz
download https://github.com/KhronosGroup/OpenCL-ICD-Loader/archive/v$OPENCL_VERSION.tar.gz OpenCL-ICD-Loader-$OPENCL_VERSION.tar.gz
download https://github.com/KhronosGroup/OpenCL-CLHPP/archive/v$CLHPP_VERSION.tar.gz OpenCL-CLHPP-$CLHPP_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xf ../OpenCL-Headers-$OPENCL_VERSION.tar.gz
tar --totals -xf ../OpenCL-ICD-Loader-$OPENCL_VERSION.tar.gz
tar --totals -xf ../OpenCL-CLHPP-$CLHPP_VERSION.tar.gz

case $PLATFORM in
    linux-arm64)
        export CC="aarch64-linux-gnu-gcc"
        export CXX="aarch64-linux-gnu-g++"
        cd OpenCL-Headers-$OPENCL_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH .
        make -j $MAKEJ
        make install/strip
        cd ../OpenCL-ICD-Loader-$OPENCL_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DOPENCL_ICD_LOADER_HEADERS_DIR=$INSTALL_PATH/include .
        make -j $MAKEJ
        make install/strip
        cd ../OpenCL-CLHPP-$CLHPP_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_PREFIX_PATH=$INSTALL_PATH -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TESTING=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        cd OpenCL-Headers-$OPENCL_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH .
        make -j $MAKEJ
        make install/strip
        cd ../OpenCL-ICD-Loader-$OPENCL_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DOPENCL_ICD_LOADER_HEADERS_DIR=$INSTALL_PATH/include .
        make -j $MAKEJ
        make install/strip
        cd ../OpenCL-CLHPP-$CLHPP_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_PREFIX_PATH=$INSTALL_PATH -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TESTING=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        cd OpenCL-Headers-$OPENCL_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH .
        make -j $MAKEJ
        make install/strip
        cd ../OpenCL-ICD-Loader-$OPENCL_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DOPENCL_ICD_LOADER_HEADERS_DIR=$INSTALL_PATH/include .
        make -j $MAKEJ
        make install/strip
        cd ../OpenCL-CLHPP-$CLHPP_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_PREFIX_PATH=$INSTALL_PATH -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TESTING=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"
        cd OpenCL-Headers-$OPENCL_VERSION
        cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH .
        ninja -j $MAKEJ
        ninja install
        cd ../OpenCL-ICD-Loader-$OPENCL_VERSION
        cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DOPENCL_ICD_LOADER_HEADERS_DIR=$INSTALL_PATH/include .
        ninja -j $MAKEJ
        ninja install
        cd ../OpenCL-CLHPP-$CLHPP_VERSION
        cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_PREFIX_PATH=$INSTALL_PATH -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TESTING=OFF .
        ninja -j $MAKEJ
        ninja install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

rm $INSTALL_PATH/include/CL/CL || true

cd ../..
