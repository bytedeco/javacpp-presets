#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" openpose
    popd
    exit
fi

OPENPOSE_VERSION=1.7.0

download https://github.com/CMU-Perceptual-Computing-Lab/openpose/archive/v$OPENPOSE_VERSION.tar.gz openpose-$OPENPOSE_VERSION.tar.gz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`

OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"
OPENCV_PATH="$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/"
CAFFE_PATH="$INSTALL_PATH/../../../caffe/cppbuild/$PLATFORM/"

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -f "$P/include/openblas_config.h" ]]; then
            OPENBLAS_PATH="$P"
        elif [[ -f "$P/include/opencv4/opencv2/core.hpp" ]]; then
            OPENCV_PATH="$P"
        elif [[ -f "$P/include/caffe/caffe.hpp" ]]; then
            CAFFE_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"
OPENCV_PATH="${OPENCV_PATH//\\//}"
CAFFE_PATH="${CAFFE_PATH//\\//}"

echo "Decompressing archives..."
tar --totals -xzf ../openpose-$OPENPOSE_VERSION.tar.gz

INSTALL_PATH=`pwd`

cd openpose-$OPENPOSE_VERSION
patch -Np1 < ../../../openpose.patch

FLAGS="-DGPU_MODE=CPU_ONLY"
if [[ "$EXTENSION" == *gpu ]]; then
    FLAGS="-DGPU_MODE=CUDA -DCMAKE_EXE_LINKER_FLAGS=\"-Wl,-rpath-link=/usr/local/cuda/lib64/stubs\" -DCUDA_ARCH=All"
fi

# We're going to get all this from Caffe's build
sedinplace '/find_package(GFlags)/d' CMakeLists.txt
sedinplace '/find_package(Glog)/d' CMakeLists.txt
sedinplace '/find_package(Protobuf REQUIRED/d' CMakeLists.txt

case $PLATFORM in
    linux-x86_64)
        CC="gcc -m64 -I/usr/local/include -I$OPENBLAS_PATH/include" CXX="g++ -m64 -I/usr/local/include -I$OPENBLAS_PATH/include" $CMAKE \
            -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
            -DCMAKE_INSTALL_LIBDIR="lib" \
            -DGFLAGS_FOUND:BOOL=ON \
            -DGFLAGS_INCLUDE_DIR=$CAFFE_PATH/include \
            -DGLOG_FOUND:BOOL=ON \
            -DGLOG_INCLUDE_DIR=$CAFFE_PATH/include \
            -DOpenCV_INCLUDE_DIRS=$OPENCV_PATH/include \
            -DOpenCV_LIBS=$OPENCV_PATH/lib/libopencv_core.so \
            -DOpenCV_LIBS=$OPENCV_PATH/lib/libopencv_highgui.so \
            -DOpenCV_LIBS=$OPENCV_PATH/lib/libopencv_imgproc.so \
            -DOpenCV_LIBS=$OPENCV_PATH/lib/libopencv_imgcodecs.so \
            -DCaffe_INCLUDE_DIRS=$CAFFE_PATH/include \
            -DCaffe_LIBS_RELEASE=$CAFFE_PATH/lib/libcaffe.so \
            -DCaffe_LIBS=$CAFFE_PATH/lib/libcaffe.so \
            -DProtobuf_INCLUDE_DIRS=$CAFFE_PATH/include \
            -DDOWNLOAD_BODY_25_MODEL=OFF \
            -DDOWNLOAD_FACE_MODEL=OFF \
            -DDOWNLOAD_HAND_MODEL=OFF \
            -DBUILD_EXAMPLES=OFF \
            -DBUILD_CAFFE=OFF \
            -DUSE_MKL=OFF \
            $FLAGS \
            -DCUDA_HOST_COMPILER="$(which g++)" \
            .
        ;;
    macosx-*)
        CC="clang -I/usr/local/include -I$OPENBLAS_PATH/include -undefined dynamic_lookup" CXX="clang++ -I/usr/local/include -I$OPENBLAS_PATH/include -undefined dynamic_lookup" $CMAKE \
            -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
            -DCMAKE_INSTALL_LIBDIR="lib" \
            -DGFLAGS_FOUND:BOOL=ON \
            -DGFLAGS_INCLUDE_DIR=$CAFFE_PATH/include \
            -DGLOG_FOUND:BOOL=ON \
            -DGLOG_INCLUDE_DIR=$CAFFE_PATH/include \
            -DOpenCV_INCLUDE_DIRS=$OPENCV_PATH/include \
            -DOpenCV_LIBS=$OPENCV_PATH/lib/libopencv_core.dylib \
            -DOpenCV_LIBS=$OPENCV_PATH/lib/libopencv_highgui.dylib \
            -DOpenCV_LIBS=$OPENCV_PATH/lib/libopencv_imgproc.dylib \
            -DOpenCV_LIBS=$OPENCV_PATH/lib/libopencv_imgcodecs.dylib \
            -DCaffe_INCLUDE_DIRS=$CAFFE_PATH/include \
            -DCaffe_LIBS_RELEASE=$CAFFE_PATH/lib/libcaffe.so \
            -DCaffe_LIBS=$CAFFE_PATH/lib/libcaffe.so \
            -DProtobuf_INCLUDE_DIRS=$CAFFE_PATH/include \
            -DDOWNLOAD_BODY_25_MODEL=OFF \
            -DDOWNLOAD_FACE_MODEL=OFF \
            -DDOWNLOAD_HAND_MODEL=OFF \
            -DBUILD_EXAMPLES=OFF \
            -DBUILD_CAFFE=OFF \
            -DUSE_MKL=OFF \
            $FLAGS \
            -DCUDA_HOST_COMPILER="$(which clang++)" \
            .
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

make -j $MAKEJ
make install

cd ../../..
