#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mxnet
    popd
    exit
fi

case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        export BLAS="openblas"
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        export BLAS="openblas"
        ;;
    macosx-*)
        export CC="clang-omp"
        export CXX="clang-omp++"
        export BLAS="apple"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

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

make -j $MAKEJ CC="$CC" CXX="$CXX" USE_BLAS="$BLAS"
cp -a include lib ../dmlc-core-$MXNET_VERSION/include ..
cp -a ../mshadow-$MXNET_VERSION/mshadow ../include

cd ../..
