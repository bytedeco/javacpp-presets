#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" opencl
    popd
    exit
fi

OPENCL_VERSION=master
download https://github.com/KhronosGroup/OpenCL-Headers/archive/$OPENCL_VERSION.tar.gz OpenCL-Headers-$OPENCL_VERSION.tar.gz
download https://github.com/KhronosGroup/OpenCL-ICD-Loader/archive/$OPENCL_VERSION.tar.gz OpenCL-ICD-Loader-$OPENCL_VERSION.tar.bz2
download https://github.com/KhronosGroup/OpenCL-CLHPP/archive/$OPENCL_VERSION.tar.gz OpenCL-CLHPP-$OPENCL_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xf ../OpenCL-Headers-$OPENCL_VERSION.tar.gz
tar --totals -xf ../OpenCL-ICD-Loader-$OPENCL_VERSION.tar.bz2
tar --totals -xf ../OpenCL-CLHPP-$OPENCL_VERSION.tar.bz2

case $PLATFORM in
    linux-x86_64)
        cd OpenCL-Headers-$OPENCL_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH .
        make -j $MAKEJ
        make install/strip
        cd ../OpenCL-ICD-Loader-$OPENCL_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DOPENCL_ICD_LOADER_HEADERS_DIR=$INSTALL_PATH/include .
        make -j $MAKEJ
        make install/strip
        cd ../OpenCL-CLHPP-$OPENCL_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DOPENCL_INCLUDE_DIR=$INSTALL_PATH/include -DOPENCL_LIB_DIR=$INSTALL_PATH/lib -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF .
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
        cd ../OpenCL-CLHPP-$OPENCL_VERSION
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DOPENCL_INCLUDE_DIR=$INSTALL_PATH/include -DOPENCL_LIB_DIR=$INSTALL_PATH/lib -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF .
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
        cd ../OpenCL-CLHPP-$OPENCL_VERSION
        cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DOPENCL_INCLUDE_DIR=$INSTALL_PATH/include -DOPENCL_LIB_DIR=$INSTALL_PATH/lib -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF .
        ninja -j $MAKEJ
        ninja install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
