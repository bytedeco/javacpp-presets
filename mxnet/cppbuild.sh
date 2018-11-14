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
export USE_MKLDNN=1
if [[ "$EXTENSION" == *gpu ]]; then
    export ADD_CFLAGS="$ADD_CFLAGS -DMXNET_USE_CUDA=1"
    export USE_CUDA=1
    export USE_CUDNN=1
    export USE_CUDA_PATH="/usr/local/cuda"
fi

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

sedinplace "s/cmake/$CMAKE/g" mkldnn.mk
sedinplace 's/kCPU/Context::kCPU/g' src/operator/tensor/elemwise_binary_scalar_op_basic.cc
sedinplace 's:../../src/operator/tensor/:./:g' src/operator/tensor/cast_storage-inl.h

sedinplace '/#include <opencv2\/opencv.hpp>/a\
#include <opencv2/imgproc/types_c.h>\
' src/io/image_augmenter.h src/io/image_io.cc tools/im2rec.cc

case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        export BLAS="openblas"
        export USE_MKLDNN=0
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
    windows-x86_64)
        # copy include files
        mkdir -p ../include
        cp -r include/mxnet 3rdparty/dmlc-core/include/dmlc 3rdparty/mshadow/mshadow ../include

        # configure the build
        mkdir -p ../build
        cd ../build
        USE_X="-DCUDA_ARCH_LIST=3.0+PTX -DUSE_CUDA=$USE_CUDA -DUSE_CUDNN=$USE_CUDNN -DUSE_OPENCV=ON -DUSE_MKLDNN=$USE_MKLDNN"
        OPENCV="-DOpenCV_DIR=$OPENCV_PATH/ -DOpenCV_CONFIG_PATH=$OPENCV_PATH/"
        OPENBLAS="-DOpenBLAS_INCLUDE_DIR=$OPENBLAS_PATH/include/ -DOpenBLAS_LIB=$OPENBLAS_PATH/lib/openblas.lib"
        "$CMAKE" -G "Visual Studio 14 2015 Win64" $USE_X $OPENCV $OPENBLAS ../apache-mxnet-src-$MXNET_VERSION-incubating

        # build the project without compiler parallelism to avoid "out of heap space"
        MSBuild.exe ALL_BUILD.vcxproj //p:Configuration=Release //p:CL_MPCount=1 //maxcpucount:$MAKEJ

        # copy binary files
        mkdir -p ../bin
        cp Release/*.dll 3rdparty/mkldnn/src/Release/*.dll ../bin

        # copy library files
        mkdir -p ../lib
        cp Release/libmxnet.lib ../lib/mxnet.lib

        # finish
        cd ../..
        return 0
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

export C_INCLUDE_PATH="$OPENBLAS_PATH/include/:$OPENCV_PATH/include/"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
export LIBRARY_PATH="$OPENBLAS_PATH/:$OPENBLAS_PATH/lib/:$OPENCV_PATH/:$OPENCV_PATH/lib/"

sed -i="" 's/$(shell pkg-config --cflags opencv)//' Makefile
sed -i="" 's/$(shell pkg-config --libs opencv)/-lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core/' Makefile
make -j $MAKEJ CC="$CC" CXX="$CXX" USE_BLAS="$BLAS" USE_OPENMP="$USE_OPENMP" CUDA_ARCH="$CUDA_ARCH" USE_CUDA="$USE_CUDA" USE_CUDNN="$USE_CUDNN" USE_CUDA_PATH="$USE_CUDA_PATH" USE_MKLDNN="$USE_MKLDNN" USE_F16C=0 ADD_CFLAGS="$ADD_CFLAGS" ADD_LDFLAGS="$ADD_LDFLAGS" lib/libmxnet.a lib/libmxnet.so
cp -r include lib 3rdparty/dmlc-core/include ..
cp -r 3rdparty/mshadow/mshadow ../include
unset CC
unset CXX

case $PLATFORM in
    macosx-*)
        install_name_tool -add_rpath @loader_path/. -id @rpath/libmxnet.so ../lib/libmxnet.so
        ;;
esac

cd ../..
