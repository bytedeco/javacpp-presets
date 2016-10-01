#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tensorflow
    popd
    exit
fi

export PYTHON_BIN_PATH=$(which python)
export TF_NEED_CUDA=0
export TF_NEED_GCP=0

TENSORFLOW_VERSION=0.10.0

download https://github.com/tensorflow/tensorflow/archive/v$TENSORFLOW_VERSION.tar.gz tensorflow-$TENSORFLOW_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM

echo "Decompressing archives"
tar --totals -xzf ../tensorflow-$TENSORFLOW_VERSION.tar.gz

# Assumes Bazel is available in the path: http://bazel.io/docs/install.html
cd tensorflow-$TENSORFLOW_VERSION


case $PLATFORM in
	android-arm)
        patch -Np1 < ../../../tensorflow-$TENSORFLOW_VERSION-android.patch
        export BUILDFLAGS=""
        ;;
    linux-x86)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        export BUILDFLAGS="--copt=-m32 --linkopt=-m32 --copt=-D_mm_cvtm64_si64=reinterpret_cast<__int64_t> --copt=-D_mm_cvtsi64_m64=reinterpret_cast<__m64>"
        ;;
    linux-x86_64)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        export BUILDFLAGS="--copt=-m64 --linkopt=-m64"
        ;;
    macosx-*)
        export BUILDFLAGS="--linkopt=-install_name --linkopt=@rpath/libtensorflow.so"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

./configure
bazel build -c opt //tensorflow:libtensorflow_cc.so $BUILDFLAGS --spawn_strategy=standalone --genrule_strategy=standalone --verbose_failures

cd ../..
