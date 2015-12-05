#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tensorflow
    popd
    exit
fi

case $PLATFORM in
    linux-x86)
        export BUILDFLAGS="--copt=-m32 --linkopt=-m32"
        ;;
    linux-x86_64)
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

PROTOBUF_VERSION=master
TENSORFLOW_VERSION=master

download https://github.com/google/protobuf/archive/$PROTOBUF_VERSION.tar.gz protobuf-$PROTOBUF_VERSION.tar.gz
download https://github.com/tensorflow/tensorflow/archive/$TENSORFLOW_VERSION.tar.gz tensorflow-$TENSORFLOW_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM

echo "Decompressing archives"
tar --totals -xzf ../protobuf-$PROTOBUF_VERSION.tar.gz
tar --totals -xzf ../tensorflow-$TENSORFLOW_VERSION.tar.gz

# Assumes Bazel is available in the path: http://bazel.io/docs/install.html
cd tensorflow-$TENSORFLOW_VERSION/google
rmdir protobuf || true
ln -snf ../../protobuf-$PROTOBUF_VERSION protobuf
cd ..
patch -Np1 < ../../../tensorflow-$TENSORFLOW_VERSION.patch
./configure
bazel build -c opt //tensorflow/cc:libtensorflow.so $BUILDFLAGS

cd ../..
