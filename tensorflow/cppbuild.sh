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
export TF_NEED_MKL=0
export TF_NEED_VERBS=0
export TF_NEED_JEMALLOC=0
export TF_NEED_CUDA=0
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_S3=0
export TF_NEED_OPENCL=0
export TF_NEED_MPI=0
export TF_NEED_GDR=0
export TF_ENABLE_XLA=0
export TF_CUDA_CLANG=0
export TF_CUDA_VERSION=9.1
export TF_CUDNN_VERSION=7
export GCC_HOST_COMPILER_PATH=$(which gcc)
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export CUDNN_INSTALL_PATH=$CUDA_TOOLKIT_PATH
export TF_CUDA_COMPUTE_CAPABILITIES=3.0

TENSORFLOW_VERSION=1.4.1

download https://github.com/tensorflow/tensorflow/archive/v$TENSORFLOW_VERSION.tar.gz tensorflow-$TENSORFLOW_VERSION.tar.gz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"

echo "Decompressing archives"
tar --totals -xzf ../tensorflow-$TENSORFLOW_VERSION.tar.gz

# Assumes Bazel is available in the path: http://bazel.io/docs/install.html
cd tensorflow-$TENSORFLOW_VERSION

# https://github.com/tensorflow/tensorflow/issues/15389
sed -i="" "s/429aa5254200/034b6c3e1017/g" tensorflow/workspace.bzl
sed -i="" "s/61d8b6fc4279dd1dda986fb1677d15e3d641c07a3ea5abe255790b1f0c0c14e9/0a8ac1e83ef9c26c0e362bd7968650b710ce54e2d883f0df84e5e45a3abe842a/g" tensorflow/workspace.bzl

export GPU_FLAGS=
if [[ "$EXTENSION" == *gpu ]]; then
    export TF_NEED_CUDA=1
    export GPU_FLAGS="--config=cuda --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma"
fi

case $PLATFORM in
    # Clang is incapable of compiling TensorFlow for Android, while in $ANDROID_NDK/source.properties,
    # the value of Pkg.Revision needs to start with "12" for Bazel to accept GCC
    android-arm)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        patch -Np1 < ../../../tensorflow-android.patch
        sed -i "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        export BUILDFLAGS="--android_compiler=gcc-4.9 --crosstool_top=//external:android/crosstool --cpu=armeabi-v7a --host_crosstool_top=@bazel_tools//tools/cpp:toolchain"
        ;;
    android-x86)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        patch -Np1 < ../../../tensorflow-android.patch
        sed -i "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        export BUILDFLAGS="--android_compiler=gcc-4.9 --crosstool_top=//external:android/crosstool --cpu=x86 --host_crosstool_top=@bazel_tools//tools/cpp:toolchain"
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
        export GCC_HOST_COMPILER_PATH=$CC
        export BUILDFLAGS="$GPU_FLAGS --copt=-m64 --linkopt=-m64"
        ;;
    macosx-*)
        # https://github.com/tensorflow/tensorflow/issues/14174
        sed -i '' 's/__align__(sizeof(T))//g' tensorflow/core/kernels/*.cu.cc

        export BUILDFLAGS="$GPU_FLAGS --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH --linkopt=-install_name --linkopt=@rpath/libtensorflow_cc.so"
        export CUDA_HOME=/usr/local/cuda
        export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
        export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
        export PATH=$DYLD_LIBRARY_PATH:$PATH
        patch -Np1 < ../../../tensorflow-macosx.patch
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

./configure
bazel build -c opt //tensorflow:libtensorflow_cc.so $BUILDFLAGS --spawn_strategy=standalone --genrule_strategy=standalone --output_filter=DONT_MATCH_ANYTHING --verbose_failures

cd ../..
