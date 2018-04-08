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
    sudo apt-get install protobuf-compiler libprotoc-dev
    pip install 'protobuf==2.6.1'
    conda install -y setuptools
    wget https://github.com/onnx/onnx/archive/v1.0.1.tar.gz
    mkdir -p $PLATFORM
    cd $PLATFORM
    tar -xzvf ../v1.0.1.tar.gz
    cd onnx-1.0.1/third_party/pybind11
    wget https://github.com/pybind/pybind11/archive/v2.2.1.tar.gz
    tar -xzvf v2.2.1.tar.gz
    mv pybind11-2.2.1/* .
    rm -r pybind11-2.2.1/
    cd ../..

    #to build with "Traditional ML" support. Untested.
    #export ONNX_ML=1
    export BASEDIR=build/temp.linux-x86_64-3.6/`pwd`/onnx/
    python setup.py install --single-version-externally-managed --record=record.txt
    g++ -v -std=c++11 -shared -Wl,-soname,libonnx.so -fPIC -o libonnx.so $BASEDIR/onnx-operators.pb.o $BASEDIR/onnx.pb.o $BASEDIR/checker.o $BASEDIR/defs/schema.o $BASEDIR/defs/tensor/old.o $BASEDIR/defs/tensor/defs.o $BASEDIR/defs/generator/defs.o $BASEDIR/defs/math/defs.o $BASEDIR/defs/data_type_utils.o $BASEDIR/defs/traditionalml/defs.o $BASEDIR/defs/experiments/defs.o $BASEDIR/defs/nn/defs.o $BASEDIR/defs/nn/old.o $BASEDIR/defs/reduction/defs.o $BASEDIR/defs/logical/defs.o $BASEDIR/defs/rnn/defs.o -pthread -lprotobuf
    cd ..
    mkdir include
    cd include
    mkdir defs
    mkdir onnx
    cd ..
    mkdir lib
    cp onnx-1.0.1/onnx/*.h include/onnx/
    cp onnx-1.0.1/onnx/defs/*.h include/defs/
    #TODO: Fix so the workaround isn't needed here
    #next line to workaround by commenting out parts of schema.h that cause failures
    patch include/defs/schema.h ../../schema.h.patch

    sudo cp onnx-1.0.1/libonnx.so /usr/lib
fi
cd ..
