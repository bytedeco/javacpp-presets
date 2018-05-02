#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" onnx
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    #No Windows support yet
    :
else

    export PROTO=2.6.1
    export INSTALL_PATH=`pwd`/$PLATFORM
    wget https://github.com/onnx/onnx/archive/v1.0.1.tar.gz

    mkdir -p $PLATFORM
    cd $PLATFORM
    wget https://github.com/google/protobuf/archive/v$PROTO.tar.gz 
    mv v$PROTO.tar.gz protobuf-$PROTO.tar.gz
    tar -xvf protobuf-$PROTO.tar.gz

    export LIBRARY_PATH=$INSTALL_PATH/lib
    export PATH=$PATH:$INSTALL_PATH/bin
    export CFLAGS="-I$INSTALL_PATH/include"
    export CXXFLAGS="-I$INSTALL_PATH/include"
    cd protobuf-$PROTO

    patch ./autogen.sh ../../../autogen.sh.patch
    ./autogen.sh
    ./configure "--prefix=$INSTALL_PATH" CFLAGS=-fPIC CXXFLAGS=-fPIC
    make -j $MAKEJ
    make install

    cd ..
    tar -xzvf ../v1.0.1.tar.gz
    cd onnx-1.0.1/third_party/pybind11
    wget https://github.com/pybind/pybind11/archive/v2.2.1.tar.gz
    tar -xzvf v2.2.1.tar.gz
    mv pybind11-2.2.1/* .
    rm -r pybind11-2.2.1/
    cd ../..

    #to build with "Traditional ML" support. Untested.
    #export ONNX_ML=1
    export BASEDIR=build/temp.linux-x86_64-?.?/`pwd`/onnx/

    python setup.py build
    g++ -v -std=c++11 -shared -Wl,-soname,libonnx.so -fPIC -o libonnx.so $BASEDIR/onnx-operators.pb.o $BASEDIR/onnx.pb.o $BASEDIR/checker.o $BASEDIR/defs/schema.o $BASEDIR/defs/tensor/old.o $BASEDIR/defs/tensor/defs.o $BASEDIR/defs/generator/defs.o $BASEDIR/defs/math/defs.o $BASEDIR/defs/data_type_utils.o $BASEDIR/defs/traditionalml/defs.o $BASEDIR/defs/experiments/defs.o $BASEDIR/defs/nn/defs.o $BASEDIR/defs/nn/old.o $BASEDIR/defs/reduction/defs.o $BASEDIR/defs/logical/defs.o $BASEDIR/defs/rnn/defs.o -pthread
    cd ..

    cd include
    mkdir defs
    mkdir onnx
    cd ..

    cp onnx-1.0.1/onnx/*.h include/onnx/
    cp onnx-1.0.1/onnx/defs/*.h include/defs/
    #TODO: Fix so the workaround isn't needed here
    #next line to workaround by commenting out parts of schema.h that cause failures
    patch include/defs/schema.h ../../schema.h.patch
    cp onnx-1.0.1/libonnx.so lib
fi
cd ..
