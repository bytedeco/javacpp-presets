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

    export PROTO=3.5.1
    export INSTALL_PATH=`pwd`/$PLATFORM
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -f -b -p $HOME/miniconda 
    export PATH="$HOME/miniconda/bin:$PATH"
    conda install -y -c conda-forge protobuf=$PROTO numpy setuptools
    export CFLAGS="-I$HOME/miniconda/include"
    export CONDA_PREFIX="$HOME/miniconda"
    wget https://github.com/onnx/onnx/archive/v1.0.1.tar.gz
    mkdir -p $PLATFORM
    cd $PLATFORM
    wget https://github.com/google/protobuf/releases/download/v$PROTO/protobuf-cpp-$PROTO.tar.gz
    mv protobuf-cpp-$PROTO.tar.gz protobuf-$PROTO.tar.gz
    tar --totals -xf protobuf-$PROTO.tar.gz
    cd protobuf-$PROTO
    ./configure "--prefix=$INSTALL_PATH" --disable-shared
    make -j `nproc`
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
    export BASEDIR=build/temp.linux-x86_64-2.7/`pwd`/onnx/
    patch setup.py ../../../setup.py.patch
    python setup.py install --single-version-externally-managed --record=record.txt
    g++ -v -std=c++0x -shared -Wl,-soname,libonnx.so -fPIC -o libonnx.so $BASEDIR/onnx-operators.pb.o $BASEDIR/onnx.pb.o $BASEDIR/checker.o $BASEDIR/defs/schema.o $BASEDIR/defs/tensor/old.o $BASEDIR/defs/tensor/defs.o $BASEDIR/defs/generator/defs.o $BASEDIR/defs/math/defs.o $BASEDIR/defs/data_type_utils.o $BASEDIR/defs/traditionalml/defs.o $BASEDIR/defs/experiments/defs.o $BASEDIR/defs/nn/defs.o $BASEDIR/defs/nn/old.o $BASEDIR/defs/reduction/defs.o $BASEDIR/defs/logical/defs.o $BASEDIR/defs/rnn/defs.o -pthread -lprotobuf
    cd ..
#    mkdir include
    cd include
    mkdir defs
    mkdir onnx
    cd ..
#    mkdir lib
    cp onnx-1.0.1/onnx/*.h include/onnx/
    cp onnx-1.0.1/onnx/defs/*.h include/defs/
    cp ~/miniconda/include/google/protobuf/message_lite.h include
    #TODO: Fix so the workaround isn't needed here
    #next line to workaround by commenting out parts of schema.h that cause failures
    patch include/defs/schema.h ../../schema.h.patch
    cp onnx-1.0.1/libonnx.so lib
fi
cd ..
