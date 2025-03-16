#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tensorflow-lite
    popd
    exit
fi

export CMAKE_FLAGS="-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
if [[ "$EXTENSION" == *gpu ]]; then
    export CMAKE_FLAGS="-DTFLITE_ENABLE_GPU=ON $CMAKE_FLAGS"
fi

TENSORFLOW_VERSION=2.19.0
download https://github.com/tensorflow/tensorflow/archive/v$TENSORFLOW_VERSION.tar.gz tensorflow-$TENSORFLOW_VERSION.tar.gz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xzf ../tensorflow-$TENSORFLOW_VERSION.tar.gz || tar --totals -xzf ../tensorflow-$TENSORFLOW_VERSION.tar.gz || true
# patch -d tensorflow-$TENSORFLOW_VERSION -Np1 < ../../tensorflow-lite.patch
# sedinplace 's/common.c/common.cc/g' tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/c/CMakeLists.txt
sedinplace 's/!defined TF_LITE_DISABLE_X86_NEON/0/g' tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/kernels/internal/optimized/neon_check.h
sedinplace 's/value = 1 << 20/value = (1 << 20)/g' tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/interpreter_options.h
sedinplace '/${TFLITE_SOURCE_DIR}\/profiling\/telemetry\/profiler.cc/a\
${TFLITE_SOURCE_DIR}\/profiling\/telemetry\/telemetry.cc\
${TFLITE_SOURCE_DIR}\/profiling\/telemetry\/c\/telemetry_setting_internal.cc\
' tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/CMakeLists.txt
sedinplace '/#include <math.h>/a\
#include <stdint.h>\
' tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/kernels/internal/spectrogram.cc

if [[ ! "$PLATFORM" == windows* ]]; then
    mkdir -p build_flatc
    cd build_flatc

    "$CMAKE" $CMAKE_FLAGS -DCMAKE_BUILD_TYPE=Release ../tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/c
    "$CMAKE" --build . --parallel $MAKEJ --target flatbuffers-flatc
    export CMAKE_FLAGS="-DTFLITE_HOST_TOOLS_DIR=$PWD/flatbuffers-flatc $CMAKE_FLAGS"

    cd ..
fi

mkdir -p build
cd build

case $PLATFORM in
    android-arm)
        export AR=ar
        export CMAKE_FLAGS="-DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 $CMAKE_FLAGS"
        ;;
    android-arm64)
        export AR=ar
        export CMAKE_FLAGS="-DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DXNNPACK_ENABLE_ARM_I8MM=OFF $CMAKE_FLAGS"
        ;;
    android-x86)
        export AR=ar
        export CMAKE_FLAGS="-DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 $CMAKE_FLAGS"
        ;;
    android-x86_64)
        export AR=ar
        export CMAKE_FLAGS="-DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DXNNPACK_ENABLE_AVXVNNI=OFF $CMAKE_FLAGS"
        ;;
    linux-armhf)
        export CC="arm-linux-gnueabihf-gcc -funsafe-math-optimizations"
        export CXX="arm-linux-gnueabihf-g++ -funsafe-math-optimizations"
        export CMAKE_FLAGS="-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=armv6 -DTFLITE_ENABLE_XNNPACK=OFF $CMAKE_FLAGS"
        ;;
    linux-arm64)
        export CC="aarch64-linux-gnu-gcc -funsafe-math-optimizations"
        export CXX="aarch64-linux-gnu-g++ -funsafe-math-optimizations"
        export CMAKE_FLAGS="-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DXNNPACK_ENABLE_ARM_I8MM=OFF $CMAKE_FLAGS"
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
        sedinplace 's/CMAKE_CXX_STANDARD 17/CMAKE_CXX_STANDARD 20/g' ../tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/CMakeLists.txt ../tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/c/CMakeLists.txt
        sedinplace 's/__PRETTY_FUNCTION__/__func__/g' ../tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/kernels/internal/optimized/depthwiseconv*.h ../tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/kernels/internal/optimized/integer_ops/depthwise_conv.h
        export CC="cl.exe -D_USE_MATH_DEFINES -DTFLITE_MMAP_DISABLED"
        export CXX="cl.exe -D_USE_MATH_DEFINES -DTFLITE_MMAP_DISABLED"
        export CMAKE_FLAGS="-G Ninja -DXNNPACK_ENABLE_AVXVNNIINT8=OFF $CMAKE_FLAGS"
        # create a dummy m.lib to satisfy some dependencies somewhere
        touch m.c
        cl.exe //c m.c
        lib m.obj
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

"$CMAKE" $CMAKE_FLAGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR=lib -DTFLITE_C_BUILD_SHARED_LIBS=OFF ../tensorflow-$TENSORFLOW_VERSION/tensorflow/lite/c
"$CMAKE" --build . --parallel $MAKEJ --target absl_log_internal_message
"$CMAKE" --build . --parallel $MAKEJ
#"$CMAKE" --install .

# since the build doesn't have an install phase, collect all object files manually
find -L $(pwd) -iname '*.obj' -o -iname '*.o' -not -path "$(pwd)/CMakeFiles/*" > objs
# remove files with main() functions as well as duplicate or unresolved symbols in them
sedinplace '/main.o/d' objs
sedinplace '/flatbuffers-flatc/d' objs
sedinplace '/CMakeCCompilerId.o/d' objs
sedinplace '/CMakeCXXCompilerId.o/d' objs
sedinplace '/tensorflowlite_c.dir/d' objs
sedinplace '/tensorflow_profiler_logger/d' objs
# convert to DOS paths with short names to prevent exceeding MAX_PATH on Windows
if which cygpath; then
    cygpath -d -f objs > objs.dos
fi

cd ../..
