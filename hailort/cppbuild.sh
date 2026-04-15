#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" hailort
    popd
    exit
fi

HAILORT_VERSION=4.23.0
download https://github.com/hailo-ai/hailort/archive/refs/tags/v$HAILORT_VERSION.tar.gz hailort-v$HAILORT_VERSION.tar.gz
mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
rm -Rf hailort-$HAILORT_VERSION
tar -xzvf ../hailort-v$HAILORT_VERSION.tar.gz
cd hailort-$HAILORT_VERSION

PATCHED_HAILORT_HEADER=hailort/libhailort/include/hailo/hailort.h
sedinplace 's/EMPTY_STRUCT_PLACEHOLDER/uint8_t reserved;/g' "$PATCHED_HAILORT_HEADER"
# MSVC cannot allocate structs with trailing zero-length arrays in generated JNI code.
sedinplace 's/hailo_detection_t detections\[0\];/hailo_detection_t detections[1];/g' "$PATCHED_HAILORT_HEADER"

case $PLATFORM in
    linux-x86_64)
        "$CMAKE" -S. -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib"
        "$CMAKE" --build build --config Release --target install --parallel $MAKEJ
        ;;
    linux-arm64)
        "$CMAKE" -S. -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib"
        "$CMAKE" --build build --config Release --target install --parallel $MAKEJ
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"
        "$CMAKE" -G "Ninja" -S. -Bbuild-ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib"
        "$CMAKE" --build build-ninja --config Release --target install --parallel $MAKEJ
        if [[ -f "$INSTALL_PATH/include/hailo/hailort.h" ]]; then
            sedinplace 's/hailo_detection_t detections\[0\];/hailo_detection_t detections[1];/g' "$INSTALL_PATH/include/hailo/hailort.h"
        fi
        cp ../bin/libhailort.dll ../lib/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
