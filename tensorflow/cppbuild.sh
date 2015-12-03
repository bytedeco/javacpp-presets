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
        export CC="gcc -m32"
        export CXX="g++ -m32"
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        ;;
    macosx-*)
        export CC="clang"
        export CXX="clang++"
        export LDFLAGS="-undefined dynamic_lookup"
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
bazel build -c opt //tensorflow/cc:libtensorflow.so

cd ../..
