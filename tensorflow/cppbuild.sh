#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tensorflow
    popd
    exit
fi

export PYTHON_BIN_PATH=$(which python)
export USE_DEFAULT_PYTHON_LIB_PATH=1
export CC_OPT_FLAGS=-O3
export TF_NEED_JEMALLOC=0
export TF_NEED_CUDA=0
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_OPENCL=0
export TF_ENABLE_XLA=0
export TF_CUDA_VERSION=8.0
export TF_CUDNN_VERSION=6
export GCC_HOST_COMPILER_PATH=$(which gcc)
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export CUDNN_INSTALL_PATH=$CUDA_TOOLKIT_PATH
export TF_CUDA_COMPUTE_CAPABILITIES=3.0

TENSORFLOW_VERSION=1.0.1

download https://github.com/tensorflow/tensorflow/archive/v$TENSORFLOW_VERSION.tar.gz tensorflow-$TENSORFLOW_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM

echo "Decompressing archives"
tar --totals -xzf ../tensorflow-$TENSORFLOW_VERSION.tar.gz

# Assumes Bazel is available in the path: http://bazel.io/docs/install.html
cd tensorflow-$TENSORFLOW_VERSION

case $PLATFORM in
    android-arm)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        patch -Np1 < ../../../tensorflow-$TENSORFLOW_VERSION-android.patch
        sed -i "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        export BUILDFLAGS="--crosstool_top=//external:android/crosstool --cpu=armeabi-v7a --host_crosstool_top=@bazel_tools//tools/cpp:toolchain"
        ;;
    android-x86)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        patch -Np1 < ../../../tensorflow-$TENSORFLOW_VERSION-android.patch
        sed -i "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        export BUILDFLAGS="--crosstool_top=//external:android/crosstool --cpu=x86 --host_crosstool_top=@bazel_tools//tools/cpp:toolchain"
        ;;
    linux-x86)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        sed -i "/        \":k8\": \[\":simd_x86_64\"\],/c\        \":k8\": \[\":simd_none\"\]," third_party/jpeg/jpeg.BUILD
        export BUILDFLAGS="--copt=-m32 --linkopt=-m32"
        ;;
    linux-x86_64)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        export TF_NEED_CUDA=1
        export GCC_HOST_COMPILER_PATH=$CC
        export BUILDFLAGS="--config=cuda --copt=-m64 --linkopt=-m64"
        patch -Np1 < ../../../tensorflow-$TENSORFLOW_VERSION-nocuda.patch
        ;;
    macosx-*)
        export TF_NEED_CUDA=1
        export BUILDFLAGS="--config=cuda --linkopt=-install_name --linkopt=@rpath/libtensorflow.so"
        patch -Np1 < ../../../tensorflow-$TENSORFLOW_VERSION-nocuda.patch
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

./configure
bazel build -c opt //tensorflow:libtensorflow_cc.so $BUILDFLAGS --spawn_strategy=standalone --genrule_strategy=standalone --output_filter=DONT_MATCH_ANYTHING --verbose_failures

case $PLATFORM in
    macosx-*)
        chmod +w bazel-bin/tensorflow/libtensorflow_cc.so
        install_name_tool -id @rpath/libtensorflow_cc.so bazel-bin/tensorflow/libtensorflow_cc.so
        ;;
esac

cd ../..
