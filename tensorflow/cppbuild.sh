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
export TF_NEED_CUDA=0
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_CUDA_VERSION=8.0
export TF_CUDNN_VERSION=5
export GCC_HOST_COMPILER_PATH=$(which gcc)
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export CUDNN_INSTALL_PATH=$CUDA_TOOLKIT_PATH
export TF_CUDA_COMPUTE_CAPABILITIES=3.0

TENSORFLOW_VERSION=0.11.0

download https://github.com/tensorflow/tensorflow/archive/v$TENSORFLOW_VERSION.tar.gz tensorflow-$TENSORFLOW_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM

echo "Decompressing archives"
tar --totals -xzf ../tensorflow-$TENSORFLOW_VERSION.tar.gz

# Assumes Bazel is available in the path: http://bazel.io/docs/install.html
cd tensorflow-$TENSORFLOW_VERSION

patch -Np1 < ../../../tensorflow-$TENSORFLOW_VERSION.patch

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
        export BUILDFLAGS="--copt=-m32 --linkopt=-m32 --copt=-D_mm_cvtm64_si64=reinterpret_cast<__int64_t> --copt=-D_mm_cvtsi64_m64=reinterpret_cast<__m64>"
        ;;
    linux-x86_64)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        export TF_NEED_CUDA=1
        export GCC_HOST_COMPILER_PATH=$CC
        export BUILDFLAGS="--config=cuda --copt=-m64 --linkopt=-m64"
        ;;
    macosx-*)
        export TF_NEED_CUDA=1
        export BUILDFLAGS="--config=cuda --linkopt=-install_name --linkopt=@rpath/libtensorflow.so"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

./configure
bazel build -c opt //tensorflow:libtensorflow_cc.so $BUILDFLAGS --spawn_strategy=standalone --genrule_strategy=standalone --verbose_failures

cd ../..
