#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tensorflow-lite
    popd
    exit
fi

export CMAKE_FLAGS=
if [[ "$EXTENSION" == *gpu ]]; then
    export CMAKE_FLAGS="-DTFLITE_ENABLE_GPU=ON"
fi

TENSORFLOW_VERSION=2.8.0
download https://github.com/tensorflow/tensorflow/archive/v$TENSORFLOW_VERSION.tar.gz tensorflow-$TENSORFLOW_VERSION.tar.gz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xzf ../tensorflow-$TENSORFLOW_VERSION.tar.gz || tar --totals -xzf ../tensorflow-$TENSORFLOW_VERSION.tar.gz
# patch -d tensorflow-$TENSORFLOW_VERSION -Np1 < ../../tensorflow-lite.patch

mkdir -p build
cd build

case $PLATFORM in
    linux-armhf)
        export CC="arm-linux-gnueabihf-gcc -march=armv6 -mfpu=vfp -funsafe-math-optimizations"
        export CXX="arm-linux-gnueabihf-g++ -march=armv6 -mfpu=vfp -funsafe-math-optimizations"
        export CMAKE_FLAGS="-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=armv6 -DTFLITE_ENABLE_XNNPACK=OFF $CMAKE_FLAGS"
        ;;
    linux-arm64)
        export CC="aarch64-linux-gnu-gcc -funsafe-math-optimizations"
        export CXX="aarch64-linux-gnu-g++ -funsafe-math-optimizations"
        export CMAKE_FLAGS="-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DTFLITE_ENABLE_XNNPACK=OFF $CMAKE_FLAGS"
        ;;
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
        ;;
    windows-x86_64)
        export CC="cl.exe -D_USE_MATH_DEFINES"
        export CXX="cl.exe -D_USE_MATH_DEFINES"
        export CMAKE_FLAGS="-G Ninja $CMAKE_FLAGS"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

"$CMAKE" $CMAKE_FLAGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR=lib ../tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/c
"$CMAKE" --build . --parallel $MAKEJ
#"$CMAKE" --install .

# since the build doesn't have an install phase, collect all object files manually
find -L $(pwd) -iname *.obj -o -iname *.o -not -path "$(pwd)/CMakeFiles/*" > objs
# remove files with main() functions as well as duplicate or unresolved symbols in them
sedinplace '/main.o/d' objs
sedinplace '/CMakeCCompilerId.o/d' objs
sedinplace '/CMakeCXXCompilerId.o/d' objs
sedinplace '/tensorflowlite_c.dir/d' objs
# convert to DOS paths with short names to prevent exceeding MAX_PATH on Windows
if which cygpath; then
    cygpath -d -f objs > objs.dos
fi

cd ../..
