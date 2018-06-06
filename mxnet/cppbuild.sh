#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mxnet
    popd
    exit
fi

export ADD_CFLAGS=
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
        export ADD_CFLAGS="-Dthread_local="
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

DLPACK_VERSION=10892ac964f1af7c81aae145cd3fab78bbccd297
DMLC_VERSION=e9446f5a53cf5e61273deff7ce814093d2791766
MSHADOW_VERSION=a8c650ce8a708608a282c4d1e251c57873a8db25
PS_VERSION=a6dda54604a07d1fb21b016ed1e3f4246b08222a
NNVM_VERSION=2bc5144cd3733fd239287e3560c7db8285d21f02
TVM_VERSION=fdba6cc9bd3bec9ccd0592fa3900b7fe25d6cb97
MXNET_VERSION=1.2.0
download https://github.com/dmlc/dlpack/archive/$DLPACK_VERSION.tar.gz dlpack-$DLPACK_VERSION.tar.gz
download https://github.com/dmlc/dmlc-core/archive/$DMLC_VERSION.tar.gz dmlc-core-$DMLC_VERSION.tar.gz
download https://github.com/dmlc/mshadow/archive/$MSHADOW_VERSION.tar.gz mshadow-$MSHADOW_VERSION.tar.gz
download https://github.com/dmlc/ps-lite/archive/$PS_VERSION.tar.gz ps-lite-$PS_VERSION.tar.gz
download https://github.com/dmlc/nnvm/archive/$NNVM_VERSION.tar.gz nnvm-$NNVM_VERSION.tar.gz
download https://github.com/dmlc/tvm/archive/$TVM_VERSION.tar.gz tvm-$TVM_VERSION.tar.gz
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
tar --totals -xzf ../tvm-$TVM_VERSION.tar.gz
tar --totals -xzf ../incubator-mxnet-$MXNET_VERSION.tar.gz
cd nnvm-$NNVM_VERSION
rmdir dmlc-core tvm || true
ln -snf ../dmlc-core-$DMLC_VERSION dmlc-core
ln -snf ../tvm-$TVM_VERSION tvm
cd ../incubator-mxnet-$MXNET_VERSION/3rdparty
rmdir dlpack dmlc-core mshadow ps-lite nnvm || true
ln -snf ../../dlpack-$DLPACK_VERSION dlpack
ln -snf ../../dmlc-core-$DMLC_VERSION dmlc-core
ln -snf ../../mshadow-$MSHADOW_VERSION mshadow
ln -snf ../../ps-lite-$PS_VERSION ps-lite
ln -snf ../../nnvm-$NNVM_VERSION nnvm
cd ..

sedinplace 's/kCPU/Context::kCPU/g' src/operator/tensor/elemwise_binary_scalar_op_basic.cc
sedinplace 's:../../src/operator/tensor/:./:g' src/operator/tensor/cast_storage-inl.h

export C_INCLUDE_PATH="$OPENBLAS_PATH/include/:$OPENCV_PATH/include/"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
export LIBRARY_PATH="$OPENBLAS_PATH/:$OPENBLAS_PATH/lib/:$OPENCV_PATH/:$OPENCV_PATH/lib/"

sed -i="" 's/$(shell pkg-config --cflags opencv)//' Makefile
sed -i="" 's/$(shell pkg-config --libs opencv)/-lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core/' Makefile
make -j $MAKEJ CC="$CC" CXX="$CXX" USE_BLAS="$BLAS" USE_OPENMP="$USE_OPENMP" USE_F16C=0 ADD_CFLAGS="-DMXNET_USE_LAPACK=1 $ADD_CFLAGS" ADD_LDFLAGS="$ADD_LDFLAGS" lib/libmxnet.a lib/libmxnet.so
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
