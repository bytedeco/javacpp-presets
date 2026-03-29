#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" onnx
    popd
    exit
fi

export MAX_JOBS=${MAKEJ:-4}

export ONNX=1.20.1
export PROTO=3.21.12

download https://github.com/onnx/onnx/archive/v$ONNX.tar.gz onnx-$ONNX.tar.gz
download https://github.com/google/protobuf/releases/download/v${PROTO:2}/protobuf-cpp-$PROTO.tar.gz protobuf-$PROTO.tar.gz

mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH="$(pwd)"

echo "Decompressing archives..."
tar --totals -xf ../onnx-$ONNX.tar.gz
tar --totals -xf ../protobuf-$PROTO.tar.gz

export LIBRARY_PATH="$INSTALL_PATH/lib"
export PATH="$INSTALL_PATH/bin:$PATH"
export CFLAGS="-I$INSTALL_PATH/include"
export CXXFLAGS="-I$INSTALL_PATH/include"
export TARGET_CMAKE_ARGS=""

case "$PLATFORM" in
    linux-arm64)
        export CC="aarch64-linux-gnu-gcc"
        export CXX="aarch64-linux-gnu-g++"
        export AR="aarch64-linux-gnu-ar"
        export RANLIB="aarch64-linux-gnu-ranlib"
        export STRIP="aarch64-linux-gnu-strip"
        export CFLAGS="$CFLAGS -fPIC"
        export CXXFLAGS="$CXXFLAGS -fPIC"
        export TARGET_CMAKE_ARGS="-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64"
        ;;
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
        export CMAKE_GENERATOR="Ninja"
        export USE_MSVC_STATIC_RUNTIME=0
        ;;
esac

####################
# Compile Protobuf #
####################
cd protobuf-$PROTO

# Build `protoc` for Host
rm -rf _build-host
mkdir -p _build-host
cd _build-host

CC= CXX= ${CMAKE:-cmake} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH/host" \
    -DCMAKE_INSTALL_LIBDIR="lib" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_BUILD_SHARED_LIBS=OFF \
    -Dprotobuf_MSVC_STATIC_RUNTIME=OFF \
    ../cmake

${CMAKE:-cmake} --build . --parallel "$MAX_JOBS" --config Release
${CMAKE:-cmake} --install . --config Release

export PROTOC="$INSTALL_PATH/host/bin/protoc"
cd ../

# Build `libprotobuf` for target
rm -rf _build-target
mkdir -p _build-target
cd _build-target

${CMAKE:-cmake} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="$CFLAGS" \
    -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
    -DCMAKE_INSTALL_LIBDIR="lib" \
    -DCMAKE_INSTALL_CMAKEDIR="cmake" \
    -Dprotobuf_MSVC_STATIC_RUNTIME=OFF \
    -Dprotobuf_BUILD_SHARED_LIBS=OFF \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_BUILD_PROTOC_BINARIES=OFF \
    -DProtobuf_PROTOC_EXECUTABLE="$PROTOC" \
    $TARGET_CMAKE_ARGS \
    ../cmake

${CMAKE:-cmake} --build . --parallel "$MAX_JOBS" --config Release
${CMAKE:-cmake} --install . --config Release
export PROTOBUF_TARGET_INCLUDE="$INSTALL_PATH/include"
export PROTOBUF_TARGET_STATIC_LIB="$INSTALL_PATH/lib/libprotobuf.a"
export PROTOBUF_TARGET_LITE_STATIC_LIB="$INSTALL_PATH/lib/libprotobuf-lite.a"
cd ../
cd ../


################
# Compile ONNX #
################
cd onnx-$ONNX

# work around issue in Xcode's version of Clang, options unsupported by Ninja, and test requirements
sedinplace 's/const std::string /std::string /g' onnx/defs/schema.h || true

rm -rf _build
mkdir -p _build
cd _build
ONNX_BUILD_PATH="$(pwd)"

${CMAKE:-cmake} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="$CFLAGS" \
    -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
    -DCMAKE_PREFIX_PATH="$INSTALL_PATH" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
    -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON \
    -DCMAKE_INSTALL_LIBDIR="lib" \
    -DONNX_ML=ON \
    -DONNX_BUILD_TESTS=OFF \
    -DONNX_WERROR=OFF \
    -DONNX_USE_LITE_PROTO=ON \
    -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF \
    -DProtobuf_DIR="$INSTALL_PATH/cmake" \
    -DProtobuf_USE_STATIC_LIBS=ON \
    -DProtobuf_PROTOC_EXECUTABLE="$PROTOC" \
    -DProtobuf_INCLUDE_DIR="$PROTOBUF_TARGET_INCLUDE" \
    -DProtobuf_LIBRARY="$PROTOBUF_TARGET_STATIC_LIB" \
    -DProtobuf_LITE_LIBRARY="$PROTOBUF_TARGET_LITE_STATIC_LIB" \
    $TARGET_CMAKE_ARGS \
    ../

${CMAKE:-cmake} --build . --parallel "$MAX_JOBS" --config Release
${CMAKE:-cmake} --install . --config Release
cd ../

mkdir -p "$INSTALL_PATH/include/onnx" \
        "$INSTALL_PATH/include/onnx/common" \
        "$INSTALL_PATH/include/onnx/defs" \
        "$INSTALL_PATH/include/onnx/version_converter" \
        "$INSTALL_PATH/include/onnx/version_converter/adapters" \
        "$INSTALL_PATH/include/onnx/shape_inference" \
        "$INSTALL_PATH/lib"

cp -a onnx/*.h "$INSTALL_PATH/include/onnx/" || true
cp -a onnx/common/*.h "$INSTALL_PATH/include/onnx/common/" || true
cp -a onnx/defs/*.h "$INSTALL_PATH/include/onnx/defs/" || true
cp -a onnx/version_converter/*.h "$INSTALL_PATH/include/onnx/version_converter/" || true
cp -a onnx/version_converter/adapters/*.h "$INSTALL_PATH/include/onnx/version_converter/adapters/" || true
cp -a onnx/shape_inference/*.h "$INSTALL_PATH/include/onnx/shape_inference/" || true

cp -a $ONNX_BUILD_PATH/*.h "$INSTALL_PATH/include/onnx/" || true
cp -a $ONNX_BUILD_PATH/libonnx* "$INSTALL_PATH/lib/" || true

cp -a $ONNX_BUILD_PATH/*.dylib "$INSTALL_PATH/lib/" || true
cp -a $ONNX_BUILD_PATH/*.so* "$INSTALL_PATH/lib/" || true
cp -a $ONNX_BUILD_PATH/*.a "$INSTALL_PATH/lib/" || true

cp -a $ONNX_BUILD_PATH/*.dll "$INSTALL_PATH/lib/" || true
cp -a $ONNX_BUILD_PATH/*.lib "$INSTALL_PATH/lib/" || true

cd ../..
