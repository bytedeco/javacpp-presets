#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mxnet
    popd
    exit
fi

MXNET_VERSION=master
download https://github.com/dmlc/dmlc-core/archive/master.tar.gz dmlc-core-$MXNET_VERSION.tar.gz
download https://github.com/dmlc/mshadow/archive/master.tar.gz mshadow-$MXNET_VERSION.tar.gz
download https://github.com/dmlc/ps-lite/archive/master.tar.gz ps-lite-$MXNET_VERSION.tar.gz
download https://github.com/dmlc/mxnet/archive/master.tar.gz mxnet-$MXNET_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
tar -xzvf ../dmlc-core-$MXNET_VERSION.tar.gz
tar -xzvf ../mshadow-$MXNET_VERSION.tar.gz
tar -xzvf ../ps-lite-$MXNET_VERSION.tar.gz
tar -xzvf ../mxnet-$MXNET_VERSION.tar.gz
cd mxnet-$MXNET_VERSION
rmdir dmlc-core mshadow ps-lite || true
ln -snf ../dmlc-core-$MXNET_VERSION dmlc-core
ln -snf ../mshadow-$MXNET_VERSION mshadow
ln -snf ../ps-lite-$MXNET_VERSION ps-lite

export PKG_CONFIG_PATH="$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/lib/pkgconfig/"

case $PLATFORM in
    linux-x86)
        make -j4 CC="gcc -m32" CXX="g++ -m32" USE_BLAS="openblas"
        cp -a include lib ../dmlc-core-$MXNET_VERSION/include ..
        cp -a ../mshadow-$MXNET_VERSION/mshadow ../include
        ;;
    linux-x86_64)
        make -j4 CC="gcc -m64" CXX="g++ -m64" USE_BLAS="openblas"
        cp -a include lib ../dmlc-core-$MXNET_VERSION/include ..
        cp -a ../mshadow-$MXNET_VERSION/mshadow ../include
        ;;
    macosx-*)
        make -j4 USE_BLAS="apple"
        cp -a include lib ../dmlc-core-$MXNET_VERSION/include ..
        cp -a ../mshadow-$MXNET_VERSION/mshadow ../include
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..

