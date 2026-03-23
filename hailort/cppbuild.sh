#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" hailort
    popd
    exit
fi

HAILORT_VERSION=5.2.0
download https://github.com/hailo-ai/hailort/archive/refs/tags/v$HAILORT_VERSION.tar.gz hailort-v$HAILORT_VERSION.tar.gz
mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
tar -xzvf ../hailort-v$HAILORT_VERSION.tar.gz
cd hailort-$HAILORT_VERSION

sedinplace 's/EMPTY_STRUCT_PLACEHOLDER/uint8_t reserved;/g' hailort/libhailort/include/hailo/hailort.h
sedinplace 's/hailo_detection_t detections\[0\];/hailo_detection_t detections[1];/g' hailort/libhailort/include/hailo/hailort.h

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
        "$CMAKE" -S. -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib"
        "$CMAKE" --build build --config Release --target install --parallel $MAKEJ
        cp ../bin/libhailort.dll ../lib/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
