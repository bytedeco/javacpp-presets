#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" openvino
    popd
    exit
fi

ADE_VERSION=0.1.1f
ONEDNN_VERSION=60f41b3a9988ce7b1bc85c4f1ce7f9443bc91c9d
GFLAGS_VERSION=2.2.1
XBYAK_VERSION=8d1e41b650890080fb77548372b6236bbd4079f9
ZLIB_VERSION=1.2.11
OPENVINO_VERSION=2021.4.2
download https://github.com/opencv/ade/archive/refs/tags/v$ADE_VERSION.tar.gz ade-v$ADE_VERSION.tar.gz
download https://github.com/openvinotoolkit/oneDNN/archive/$ONEDNN_VERSION.tar.gz oneDNN-$ONEDNN_VERSION.tar.gz
download https://github.com/gflags/gflags/archive/refs/tags/v${GFLAGS_VERSION}.tar.gz gflags-v${GFLAGS_VERSION}.tar.gz
download https://github.com/herumi/xbyak/archive/$XBYAK_VERSION.tar.gz xbyak-$XBYAK_VERSION.tar.gz
download http://zlib.net/zlib-$ZLIB_VERSION.tar.gz zlib-$ZLIB_VERSION.tar.gz
download https://github.com/openvinotoolkit/openvino/archive/refs/tags/$OPENVINO_VERSION.tar.gz openvino-$OPENVINO_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo $INSTALL_PATH
echo "Decompressing archives..."
tar --totals -xzf ../openvino-$OPENVINO_VERSION.tar.gz
# cp -r openvino-$OPENVINO_VERSION openvino-$OPENVINO_VERSION.orig
patch -Np1 -d openvino-$OPENVINO_VERSION < ../../openvino.patch
cd openvino-$OPENVINO_VERSION
tar --totals --strip-components=1 -xzf ../../ade-v$ADE_VERSION.tar.gz --directory inference-engine/thirdparty/ade
tar --totals --strip-components=1 -xzf ../../oneDNN-$ONEDNN_VERSION.tar.gz --directory inference-engine/thirdparty/mkl-dnn
tar --totals --strip-components=1 -xzf ../../gflags-v$GFLAGS_VERSION.tar.gz --directory inference-engine/samples/thirdparty/gflags
tar --totals --strip-components=1 -xzf ../../xbyak-$XBYAK_VERSION.tar.gz --directory thirdparty/xbyak
tar --totals --strip-components=1 -xzf ../../zlib-$ZLIB_VERSION.tar.gz --directory thirdparty/zlib/zlib


case $PLATFORM in
    linux-x86_64)
        CC="gcc-9 -m64" CXX="g++-9 -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DENABLE_OPENCV=OFF -DENABLE_CLANG_FORMAT=OFF -DENABLE_SPEECH_DEMO=OFF -DENABLE_SAMPLES=OFF -DENABLE_CPPLINT=OFF -DBUILD_SHARED_LIBS=ON .
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
