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

    export PROTO=3.3.2
    export ONNX=1.2.2
    export INSTALL_PATH=`pwd`/$PLATFORM
    wget https://github.com/onnx/onnx/archive/v$ONNX.tar.gz

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

    ./autogen.sh
    ./configure "--prefix=$INSTALL_PATH" CFLAGS=-fPIC CXXFLAGS=-fPIC
    make -j $MAKEJ
    make install

    cd ..
    tar -xzvf ../v$ONNX.tar.gz
    cd onnx-$ONNX/third_party/pybind11
    wget https://github.com/pybind/pybind11/archive/v2.2.1.tar.gz
    tar -xzvf v2.2.1.tar.gz
    mv pybind11-2.2.1/* .
    rm -r pybind11-2.2.1/
    cd ../..

    #to build with "Traditional ML" support. Untested.
    #export ONNX_ML=1
    export DEFSBASEDIR=.setuptools-cmake-build/CMakeFiles/onnx.dir/onnx
    export PROTOBASEDIR=.setuptools-cmake-build/CMakeFiles/onnx_proto.dir/onnx/
    export INCLUDEBASEDIR=.setuptools-cmake-build/onnx/

    python3 setup.py build
    g++ -v -std=c++11 -shared -Wl,-soname,libonnx.so -fPIC -DONNX_NAMESPACE=onnx -o libonnx.so $PROTOBASEDIR/onnx-operators.pb.cc.o $PROTOBASEDIR/onnx.pb.cc.o $DEFSBASEDIR/checker.cc.o $DEFSBASEDIR/defs/schema.cc.o $DEFSBASEDIR/defs/tensor/old.cc.o $DEFSBASEDIR/defs/tensor/defs.cc.o $DEFSBASEDIR/defs/generator/defs.cc.o $DEFSBASEDIR/defs/math/defs.cc.o $DEFSBASEDIR/defs/data_type_utils.cc.o $DEFSBASEDIR/defs/traditionalml/defs.cc.o $DEFSBASEDIR/defs/experiments/defs.cc.o $DEFSBASEDIR/defs/nn/defs.cc.o $DEFSBASEDIR/defs/nn/old.cc.o $DEFSBASEDIR/defs/reduction/defs.cc.o $DEFSBASEDIR/defs/logical/defs.cc.o $DEFSBASEDIR/defs/rnn/defs.cc.o -pthread
    cd ..

    cd include
    mkdir defs
    mkdir onnx
    cd onnx
    mkdir defs
    mkdir common
    cd ..
    cd ..

    cp onnx-$ONNX/onnx/*.h include/onnx/
    cp onnx-$ONNX/onnx/common/*.h include/onnx/common/
    cp onnx-$ONNX/onnx/defs/shape_inference.h include/onnx/defs/
    cp onnx-$ONNX/onnx/defs/data_type_utils.h include/onnx/defs/
    cp onnx-$ONNX/$INCLUDEBASEDIR/*.h include/onnx/
    cp onnx-$ONNX/onnx/defs/*.h include/defs/
    cd include
    grep -rl ONNX_NAMESPACE . | xargs sed -i 's/ONNX_NAMESPACE/onnx/g'
    cd ..
    #TODO: Fix so the workaround isn't needed here
    #next line to workaround by commenting out parts of schema.h that cause failures
    patch include/defs/schema.h ../../schema.h.patch
    cp onnx-$ONNX/libonnx.so lib
fi
cd ..
