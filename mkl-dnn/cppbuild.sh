#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mkl-dnn
    popd
    exit
fi

MKLDNN_VERSION=1.0
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
        export CC="$(ls -1 /usr/local/bin/gcc-? | head -n 1)"
        export CXX="$(ls -1 /usr/local/bin/g++-? | head -n 1)"
        sedinplace 's/__thread/thread_local/g' src/common/utils.hpp
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DARCH_OPT_FLAGS='' -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        unset CC
        unset CXX
        ;;
    windows-x86_64)
        "$CMAKE" -G "MSYS Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DARCH_OPT_FLAGS='' -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
