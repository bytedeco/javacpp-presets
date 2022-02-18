#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" onnx
    popd
    exit
fi

# to build with "Traditional ML" support. Untested.
export ONNX_ML=1
export CMAKE_BUILD_DIR=.setuptools-cmake-build/
export CMAKE_ARGS=-DBUILD_SHARED_LIBS=ON
export MAX_JOBS=$MAKEJ

export ONNX=1.11.0
export PROTO=3.11.3
export PYBIND=2.6.2

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
export PYTHON_BIN_PATH=$(which python3)

case $PLATFORM in
    linux-*)
        export CC="gcc"
        export CXX="g++"
        export CFLAGS="$CFLAGS -fPIC"
        export CXXFLAGS="$CXXFLAGS -fPIC"
        ;;
    macosx-*)
        export CC="clang"
        export CXX="clang++"
        export CFLAGS="$CFLAGS -fPIC"
        export CXXFLAGS="$CXXFLAGS -fPIC"
        ;;
    windows-*)
        export CC="cl.exe"
        export CXX="cl.exe"
        export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=OFF -DProtobuf_USE_STATIC_LIBS=ON"
        export CMAKE_GENERATOR="Ninja"
        export PYTHON_BIN_PATH=$(which python.exe)
        export USE_MSVC_STATIC_RUNTIME=0
        ;;
esac

cd protobuf-$PROTO
"$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF cmake
"$CMAKE" --build . --parallel $MAKEJ --config Release
"$CMAKE" --install . --config Release

cd ../onnx-$ONNX
rm -Rf third_party/pybind11
ln -sf $INSTALL_PATH/pybind11-$PYBIND third_party/pybind11
# work around issue in Xcode's version of Clang, options unsupported by Ninja, and test requirements
sedinplace 's/const std::string /std::string /g' onnx/defs/schema.h
sedinplace 's/if WINDOWS:/if False:/g' setup.py
sedinplace '/if platform.architecture/{N;N;N;d;}' setup.py
sedinplace "/setup_requires.append('pytest-runner')/d" setup.py
"$PYTHON_BIN_PATH" setup.py --quiet build

mkdir -p ../include/onnx ../include/onnx/common ../include/onnx/defs ../include/onnx/optimizer/ ../include/onnx/optimizer/passes ../include/onnx/version_converter ../include/onnx/version_converter/adapters ../include/onnx/shape_inference

cp onnx/*.h ../include/onnx/
cp onnx/common/*.h ../include/onnx/common/
cp onnx/defs/*.h ../include/onnx/defs/
#cp onnx/optimizer/*.h ../include/onnx/optimizer/
#cp onnx/optimizer/passes/*.h ../include/onnx/optimizer/passes/
cp onnx/version_converter/*.h ../include/onnx/version_converter/
cp onnx/version_converter/adapters/*.h ../include/onnx/version_converter/adapters/
cp onnx/shape_inference/*.h ../include/onnx/shape_inference/
cp $CMAKE_BUILD_DIR/onnx/*.h ../include/onnx/
cp $CMAKE_BUILD_DIR/libonnx* ../lib || true
cp $CMAKE_BUILD_DIR/onnx*.{lib,dll} ../lib || true

cd ../..
