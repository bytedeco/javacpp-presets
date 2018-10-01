#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mxnet
    popd
    exit
fi

export ADD_CFLAGS="-DMXNET_USE_LAPACK=1"
export ADD_LDFLAGS=
export USE_OPENMP=1
export CUDA_ARCH=-arch=sm_30
export USE_CUDA=0
export USE_CUDNN=0
export USE_CUDA_PATH=
if [[ "$EXTENSION" == *gpu ]]; then
    export ADD_CFLAGS="$ADD_CFLAGS -DMXNET_USE_CUDA=1"
    export USE_CUDA=1
    export USE_CUDNN=1
    export USE_CUDA_PATH="/usr/local/cuda"
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
        if which g++-6 &> /dev/null; then
            export CC="gcc-6 -m64"
            export CXX="g++-6 -m64"
        fi
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

MXNET_VERSION=1.3.0
download http://apache.org/dist/incubator/mxnet/$MXNET_VERSION/apache-mxnet-src-$MXNET_VERSION-incubating.tar.gz apache-mxnet-src-$MXNET_VERSION-incubating.tar.gz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
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
tar --totals -xzf ../apache-mxnet-src-$MXNET_VERSION-incubating.tar.gz

cd apache-mxnet-src-$MXNET_VERSION-incubating

sedinplace 's/kCPU/Context::kCPU/g' src/operator/tensor/elemwise_binary_scalar_op_basic.cc
sedinplace 's:../../src/operator/tensor/:./:g' src/operator/tensor/cast_storage-inl.h

export C_INCLUDE_PATH="$OPENBLAS_PATH/include/:$OPENCV_PATH/include/"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
export LIBRARY_PATH="$OPENBLAS_PATH/:$OPENBLAS_PATH/lib/:$OPENCV_PATH/:$OPENCV_PATH/lib/"

sed -i="" 's/$(shell pkg-config --cflags opencv)//' Makefile
sed -i="" 's/$(shell pkg-config --libs opencv)/-lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core/' Makefile
make -j $MAKEJ CC="$CC" CXX="$CXX" USE_BLAS="$BLAS" USE_OPENMP="$USE_OPENMP" CUDA_ARCH="$CUDA_ARCH" USE_CUDA="$USE_CUDA" USE_CUDNN="$USE_CUDNN" USE_CUDA_PATH="$USE_CUDA_PATH" USE_F16C=0 ADD_CFLAGS="$ADD_CFLAGS" ADD_LDFLAGS="$ADD_LDFLAGS" lib/libmxnet.a lib/libmxnet.so
cp -a include lib 3rdparty/dmlc-core/include ..
cp -a 3rdparty/mshadow/mshadow ../include
unset CC
unset CXX

case $PLATFORM in
    macosx-*)
        install_name_tool -add_rpath @loader_path/. -id @rpath/libmxnet.so ../lib/libmxnet.so
        ;;
esac

cd ../..
