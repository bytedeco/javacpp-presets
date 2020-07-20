#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" dnnl
    popd
    exit
fi

TBB_VERSION=2020.3
MKLDNN_VERSION=1.5.1
download https://github.com/oneapi-src/oneTBB/archive/v$TBB_VERSION.tar.gz oneTBB-$TBB_VERSION.tar.bz2
download https://github.com/oneapi-src/oneDNN/archive/v$MKLDNN_VERSION.tar.gz oneDNN-$MKLDNN_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib bin
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xf ../oneTBB-$TBB_VERSION.tar.bz2
tar --totals -xf ../oneDNN-$MKLDNN_VERSION.tar.bz2
cd oneDNN-$MKLDNN_VERSION
patch -Np1 < ../../../mkl-dnn.patch

sedinplace 's/-fvisibility=internal//g' cmake/platform.cmake
sedinplace 's/-fvisibility-inlines-hidden//g' cmake/platform.cmake

case $PLATFORM in
    linux-x86_64)
        cd ../oneTBB-$TBB_VERSION
        make -j $MAKEJ tbb_os=linux
        sedinplace 's/release/debug/g' Makefile
        make -j $MAKEJ tbb_os=linux
        cp -a include/* ../include
        cp -a build/*release/libtbb.* ../lib
        cp -a build/*debug/libtbb_debug.* ../lib
        strip ../lib/libtbb.so.*
        cd ../oneDNN-$MKLDNN_VERSION
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' -DARCH_OPT_FLAGS='-Wno-error' -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF -DDNNL_CPU_RUNTIME="TBB" -DTBBROOT=$INSTALL_PATH .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        cd ../oneTBB-$TBB_VERSION
        make -j $MAKEJ tbb_os=macos
        sedinplace 's/release/debug/g' Makefile
        make -j $MAKEJ tbb_os=macos
        cp -a include/* ../include
        cp -a build/*release/libtbb.* ../lib
        cp -a build/*debug/libtbb_debug.* ../lib
        cd ../oneDNN-$MKLDNN_VERSION
        sedinplace 's/__thread/thread_local/g' src/common/utils.hpp
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DARCH_OPT_FLAGS='' -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF -DDNNL_CPU_RUNTIME="TBB" -DTBBROOT=$INSTALL_PATH .
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86_64)
        cd ../oneTBB-$TBB_VERSION
        patch -Np1 < ../../../tbb-windows.patch
        make -j $MAKEJ tbb_os=windows runtime=vc14
        sedinplace 's/release/debug/g' Makefile
        make -j $MAKEJ tbb_os=windows runtime=vc14
        mkdir -p ../lib/intel64/vc14/
        cp -a include/* ../include
        cp -a build/*release/tbb.dll ../lib/
        cp -a build/*release/tbb.lib ../lib/intel64/vc14/
        cp -a build/*debug/tbb_debug.lib ../lib/intel64/vc14/
        cd ../oneDNN-$MKLDNN_VERSION
        "$CMAKE" -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DARCH_OPT_FLAGS='' -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF -DDNNL_CPU_RUNTIME="TBB" -DTBBROOT=$INSTALL_PATH .
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
