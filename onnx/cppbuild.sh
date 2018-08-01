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
    echo "Error: Platform \"$PLATFORM\" is not supported"
    exit 1
fi

export ONNX=1.2.2
export PROTO=3.5.1
export PYBIND=2.2.3

download https://github.com/onnx/onnx/archive/v$ONNX.tar.gz onnx-$ONNX.tar.gz
download https://github.com/google/protobuf/releases/download/v$PROTO/protobuf-cpp-$PROTO.tar.gz protobuf-$PROTO.tar.gz
download https://github.com/pybind/pybind11/archive/v$PYBIND.tar.gz pybind11-$PYBIND.tar.gz

mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xf ../onnx-$ONNX.tar.gz
tar --totals -xf ../protobuf-$PROTO.tar.gz
tar --totals -xf ../pybind11-$PYBIND.tar.gz

export LIBRARY_PATH="$INSTALL_PATH/lib"
export PATH="$PATH:$INSTALL_PATH/bin"
export CFLAGS="-I$INSTALL_PATH/include"
export CXXFLAGS="-I$INSTALL_PATH/include"

cd protobuf-$PROTO
./configure "--prefix=$INSTALL_PATH" CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared
make -j $MAKEJ V=0
make install

cd ../onnx-$ONNX
rm -df third_party/pybind11
ln -sf $INSTALL_PATH/pybind11-$PYBIND third_party/pybind11

#to build with "Traditional ML" support. Untested.
#export ONNX_ML=1
export DEFSBASEDIR=.setuptools-cmake-build/CMakeFiles/onnx.dir/onnx
export PROTOBASEDIR=.setuptools-cmake-build/CMakeFiles/onnx_proto.dir/onnx/
export INCLUDEBASEDIR=.setuptools-cmake-build/onnx/

python3 setup.py build
g++ -v -std=c++11 -shared -Wl,-soname,libonnx.so -fPIC -DONNX_NAMESPACE=onnx -o libonnx.so $PROTOBASEDIR/onnx-operators.pb.cc.o $PROTOBASEDIR/onnx.pb.cc.o $DEFSBASEDIR/checker.cc.o $DEFSBASEDIR/defs/schema.cc.o $DEFSBASEDIR/defs/tensor/old.cc.o $DEFSBASEDIR/defs/tensor/defs.cc.o $DEFSBASEDIR/defs/generator/defs.cc.o $DEFSBASEDIR/defs/math/defs.cc.o $DEFSBASEDIR/defs/data_type_utils.cc.o $DEFSBASEDIR/defs/traditionalml/defs.cc.o $DEFSBASEDIR/defs/experiments/defs.cc.o $DEFSBASEDIR/defs/nn/defs.cc.o $DEFSBASEDIR/defs/nn/old.cc.o $DEFSBASEDIR/defs/reduction/defs.cc.o $DEFSBASEDIR/defs/logical/defs.cc.o $DEFSBASEDIR/defs/rnn/defs.cc.o -lprotobuf -pthread

mkdir -p ../include/onnx ../include/onnx/common ../include/onnx/defs
cp onnx/*.h ../include/onnx/
cp $INCLUDEBASEDIR/*.h ../include/onnx/
cp onnx/common/*.h ../include/onnx/common/
cp onnx/defs/*.h ../include/onnx/defs/
cp libonnx.so ../lib

cd ../..
