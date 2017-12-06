#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mxnet
    popd
    exit
fi

export ADD_LDFLAGS=
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
        export CC="$(ls -1 /usr/local/bin/gcc-? | head -n 1)"
        export CXX="$(ls -1 /usr/local/bin/g++-? | head -n 1)"
        export BLAS="openblas"
        export ADD_LDFLAGS="-static-libgcc -static-libstdc++"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

DLPACK_VERSION=a6e09b58dc00ee0065f5b7879800e646fbb01d1e
DMLC_VERSION=87b7ffa59eb78f753073ac56f5f60e46d930b93c
MSHADOW_VERSION=2d7780c3f2eefe4453fa419862d1b2089bedb8d5
PS_VERSION=2ce8b9a256207947acfa2cb9b09ab74b8de74547
NNVM_VERSION=e4a138ab947d682c83625840bbcd66f70feb4b14
MXNET_VERSION=1.0.0
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

export C_INCLUDE_PATH="$OPENBLAS_PATH/include/:$OPENCV_PATH/include/"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
export LIBRARY_PATH="$OPENBLAS_PATH/:$OPENBLAS_PATH/lib/:$OPENCV_PATH/:$OPENCV_PATH/lib/"

sed -i="" 's/$(shell pkg-config --cflags opencv)//' Makefile
sed -i="" 's/$(shell pkg-config --libs opencv)/-lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core/' Makefile
make -j $MAKEJ CC="$CC" CXX="$CXX" USE_BLAS="$BLAS" ADD_CFLAGS="-DMXNET_USE_LAPACK -fopenmp" ADD_LDFLAGS="$ADD_LDFLAGS" lib/libmxnet.a lib/libmxnet.so
cp -a include lib ../dmlc-core-$DMLC_VERSION/include ..
cp -a ../mshadow-$MSHADOW_VERSION/mshadow ../include

case $PLATFORM in
    macosx-*)
        install_name_tool -add_rpath @loader_path/. -id @rpath/libmxnet.so ../lib/libmxnet.so
        ;;
esac

cd ../..
