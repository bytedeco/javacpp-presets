#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mkl-dnn
    popd
    exit
fi

MKLDNN_VERSION=0.15
MKLML_VERSION=2018.0.3.20180406
download https://github.com/intel/mkl-dnn/archive/v$MKLDNN_VERSION.tar.gz mkl-dnn-$MKLDNN_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xf ../mkl-dnn-$MKLDNN_VERSION.tar.bz2
cd mkl-dnn-$MKLDNN_VERSION

sedinplace 's/-fvisibility=internal//g' cmake/platform.cmake
sedinplace 's/-fvisibility-inlines-hidden//g' cmake/platform.cmake

case $PLATFORM in
    linux-x86_64)
        download https://github.com/intel/mkl-dnn/releases/download/v$MKLDNN_VERSION/mklml_lnx_$MKLML_VERSION.tgz mklml_lnx_$MKLML_VERSION.tgz
        mkdir -p external
        tar --totals -xf mklml_lnx_$MKLML_VERSION.tgz -C external
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -Wno-error=unused-result
        make -j $MAKEJ
        make install/strip
        cp external/mklml_lnx_$MKLML_VERSION/include/* ../include/
        ;;
    macosx-x86_64)
        download https://github.com/intel/mkl-dnn/releases/download/v$MKLDNN_VERSION/mklml_mac_$MKLML_VERSION.tgz mklml_mac_$MKLML_VERSION.tgz
        mkdir -p external
        tar --totals -xf mklml_mac_$MKLML_VERSION.tgz -C external
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH
        make -j $MAKEJ
        make install/strip
        cp external/mklml_mac_$MKLML_VERSION/include/* ../include/
        ;;
    windows-x86_64)
        download https://github.com/intel/mkl-dnn/releases/download/v$MKLDNN_VERSION/mklml_win_$MKLML_VERSION.zip mklml_win_$MKLML_VERSION.zip
        mkdir -p external
        unzip -o mklml_win_$MKLML_VERSION.zip -d external
        "$CMAKE" -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //maxcpucount:$MAKEJ
        cp external/mklml_win_$MKLML_VERSION/include/* ../include/
        cp external/mklml_win_$MKLML_VERSION/lib/*.lib ../lib/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
