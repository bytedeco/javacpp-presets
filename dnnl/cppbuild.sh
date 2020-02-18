#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" dnnl
    popd
    exit
fi

MKLDNN_VERSION=1.2
download https://github.com/intel/mkl-dnn/archive/v$MKLDNN_VERSION.tar.gz mkl-dnn-$MKLDNN_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xf ../mkl-dnn-$MKLDNN_VERSION.tar.bz2
cd mkl-dnn-$MKLDNN_VERSION
patch -Np1 < ../../../mkl-dnn.patch

sedinplace 's/-fvisibility=internal//g' cmake/platform.cmake
sedinplace 's/-fvisibility-inlines-hidden//g' cmake/platform.cmake

case $PLATFORM in
    linux-x86_64)
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' -DARCH_OPT_FLAGS='-Wno-error' -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        sedinplace 's/__thread/thread_local/g' src/common/utils.hpp
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DARCH_OPT_FLAGS='' -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86_64)
        "$CMAKE" -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DARCH_OPT_FLAGS='' -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF .
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
