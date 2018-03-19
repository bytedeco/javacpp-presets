#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mxnet
    popd
    exit
fi

export ADD_LDFLAGS=
export USE_OPENMP=1
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
        export CC="clang"
        export CXX="clang++"
        export BLAS="openblas"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

DLPACK_VERSION=10892ac964f1af7c81aae145cd3fab78bbccd297
DMLC_VERSION=282b98663f59df6b26f906580af610dea3046f22
MSHADOW_VERSION=f5b67f380cb0588be11e6f440f92f013139380ee
PS_VERSION=aee325276bccb092f516df0bce30d3a8333f4038
NNVM_VERSION=c342da72271c85e477480323f1d91997c6101ac0
MXNET_VERSION=1.1.0
download https://github.com/dmlc/dlpack/archive/$DLPACK_VERSION.tar.gz dlpack-$DLPACK_VERSION.tar.gz
download https://github.com/dmlc/dmlc-core/archive/$DMLC_VERSION.tar.gz dmlc-core-$DMLC_VERSION.tar.gz
download https://github.com/dmlc/mshadow/archive/$MSHADOW_VERSION.tar.gz mshadow-$MSHADOW_VERSION.tar.gz
download https://github.com/dmlc/ps-lite/archive/$PS_VERSION.tar.gz ps-lite-$PS_VERSION.tar.gz
download https://github.com/dmlc/nnvm/archive/$NNVM_VERSION.tar.gz nnvm-$NNVM_VERSION.tar.gz
download https://github.com/apache/incubator-mxnet/archive/$MXNET_VERSION.tar.gz incubator-mxnet-$MXNET_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

OPENCV_PATH="$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/"
OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -d "$P/include/opencv2" ]]; then
            OPENCV_PATH="$P"
        elif [[ -f "$P/include/openblas_config.h" ]]; then
            OPENBLAS_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

echo "Decompressing archives..."
tar --totals -xzf ../dlpack-$DLPACK_VERSION.tar.gz
tar --totals -xzf ../dmlc-core-$DMLC_VERSION.tar.gz
tar --totals -xzf ../mshadow-$MSHADOW_VERSION.tar.gz
tar --totals -xzf ../ps-lite-$PS_VERSION.tar.gz
tar --totals -xzf ../nnvm-$NNVM_VERSION.tar.gz
tar --totals -xzf ../incubator-mxnet-$MXNET_VERSION.tar.gz
cd incubator-mxnet-$MXNET_VERSION
rmdir dlpack dmlc-core mshadow ps-lite nnvm || true
ln -snf ../dlpack-$DLPACK_VERSION dlpack
ln -snf ../dmlc-core-$DMLC_VERSION dmlc-core
ln -snf ../mshadow-$MSHADOW_VERSION mshadow
ln -snf ../ps-lite-$PS_VERSION ps-lite
ln -snf ../nnvm-$NNVM_VERSION nnvm

sed -i="" 's/kCPU/Context::kCPU/g' src/operator/tensor/elemwise_binary_scalar_op_basic.cc

export C_INCLUDE_PATH="$OPENBLAS_PATH/include/:$OPENCV_PATH/include/"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
export LIBRARY_PATH="$OPENBLAS_PATH/:$OPENBLAS_PATH/lib/:$OPENCV_PATH/:$OPENCV_PATH/lib/:$OPENCV_PATH/lib64/"

sed -i="" 's/$(shell pkg-config --cflags opencv)//' Makefile
sed -i="" 's/$(shell pkg-config --libs opencv)/-lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core/' Makefile
make -j $MAKEJ CC="$CC" CXX="$CXX" USE_BLAS="$BLAS" USE_OPENMP="$USE_OPENMP" ADD_CFLAGS="-DMXNET_USE_LAPACK" ADD_LDFLAGS="$ADD_LDFLAGS" lib/libmxnet.a lib/libmxnet.so
cp -a include lib ../dmlc-core-$DMLC_VERSION/include ..
cp -a ../mshadow-$MSHADOW_VERSION/mshadow ../include
unset CC
unset CXX

case $PLATFORM in
    macosx-*)
        install_name_tool -add_rpath @loader_path/. -id @rpath/libmxnet.so ../lib/libmxnet.so
        ;;
esac

cd ../..
