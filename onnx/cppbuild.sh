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

export ONNX=1.8.1
export PROTO=3.7.1
export PYBIND=2.6.0

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
export PATH="$INSTALL_PATH/bin:$PATH"
export CFLAGS="-I$INSTALL_PATH/include"
export CXXFLAGS="-I$INSTALL_PATH/include"

cd protobuf-$PROTO
./configure "--prefix=$INSTALL_PATH" CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared
make -j $MAKEJ V=0
make install

cd ../onnx-$ONNX
rm -Rf third_party/pybind11
ln -sf $INSTALL_PATH/pybind11-$PYBIND third_party/pybind11
# work around issue in Xcode's version of Clang
sedinplace 's/const std::string /std::string /g' onnx/defs/schema.h

#to build with "Traditional ML" support. Untested.
export ONNX_ML=1
export CMAKE_BUILD_DIR=.setuptools-cmake-build/
export CMAKE_ARGS=-DBUILD_SHARED_LIBS=ON
python3 setup.py --quiet build

mkdir -p ../include/onnx ../include/onnx/common ../include/onnx/defs ../include/onnx/optimizer/ ../include/onnx/optimizer/passes ../include/onnx/version_converter ../include/onnx/version_converter/adapters ../include/onnx/shape_inference

cp onnx/*.h ../include/onnx/
cp onnx/common/*.h ../include/onnx/common/
cp onnx/defs/*.h ../include/onnx/defs/
cp onnx/optimizer/*.h ../include/onnx/optimizer/
cp onnx/optimizer/passes/*.h ../include/onnx/optimizer/passes/
cp onnx/version_converter/*.h ../include/onnx/version_converter/
cp onnx/version_converter/adapters/*.h ../include/onnx/version_converter/adapters/
cp onnx/shape_inference/*.h ../include/onnx/shape_inference/
cp $CMAKE_BUILD_DIR/onnx/*.h ../include/onnx/
cp $CMAKE_BUILD_DIR/libonnx* ../lib

cd ../..
