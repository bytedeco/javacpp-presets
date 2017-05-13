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

MXNET_VERSION=master
download https://github.com/dmlc/dmlc-core/archive/master.tar.gz dmlc-core-$MXNET_VERSION.tar.gz
download https://github.com/dmlc/mshadow/archive/master.tar.gz mshadow-$MXNET_VERSION.tar.gz
download https://github.com/dmlc/ps-lite/archive/master.tar.gz ps-lite-$MXNET_VERSION.tar.gz
download https://github.com/dmlc/nnvm/archive/master.tar.gz nnvm-$MXNET_VERSION.tar.gz
download https://github.com/dmlc/mxnet/archive/master.tar.gz mxnet-$MXNET_VERSION.tar.gz

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
tar --totals -xzf ../dmlc-core-$MXNET_VERSION.tar.gz
tar --totals -xzf ../mshadow-$MXNET_VERSION.tar.gz
tar --totals -xzf ../ps-lite-$MXNET_VERSION.tar.gz
tar --totals -xzf ../nnvm-$MXNET_VERSION.tar.gz
tar --totals -xzf ../mxnet-$MXNET_VERSION.tar.gz
cd mxnet-$MXNET_VERSION
rmdir dmlc-core mshadow ps-lite nnvm || true
ln -snf ../dmlc-core-$MXNET_VERSION dmlc-core
ln -snf ../mshadow-$MXNET_VERSION mshadow
ln -snf ../ps-lite-$MXNET_VERSION ps-lite
ln -snf ../nnvm-$MXNET_VERSION nnvm

export C_INCLUDE_PATH="$OPENBLAS_PATH/include/:$OPENCV_PATH/include/"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
export LIBRARY_PATH="$OPENBLAS_PATH/lib/:$OPENCV_PATH/lib/"

sed -i="" 's/$(shell pkg-config --cflags opencv)//' Makefile
sed -i="" 's/$(shell pkg-config --libs opencv)/-lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core/' Makefile
make -j $MAKEJ CC="$CC" CXX="$CXX" USE_BLAS="$BLAS" ADD_LDFLAGS="$ADD_LDFLAGS" lib/libmxnet.a lib/libmxnet.so
cp -a include lib ../dmlc-core-$MXNET_VERSION/include ..
cp -a ../mshadow-$MXNET_VERSION/mshadow ../include

case $PLATFORM in
    macosx-*)
        install_name_tool -add_rpath @loader_path/. -id @rpath/libmxnet.so ../lib/libmxnet.so
        ;;
esac

cd ../..
