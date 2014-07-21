#!/bin/sh
if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

# FLANDMARK_VERSION=a0981a3b09cc5534255dc1dcdae2179097231bdd
# download https://github.com/uricamic/flandmark/archive/$FLANDMARK_VERSION.zip flandmark.zip

# unzip -o flandmark.zip -d flandmark/sources

mkdir -p $PLATFORM
cd $PLATFORM

mkdir -p include
cp -r ../flandmark/sources/libflandmark/flandmark_detector.h include
cp -r ../flandmark/sources/libflandmark/msvc-compat.h include
cp -r ../../../opencv/cppbuild/$PLATFORM/include/opencv/cv.h include
cp -r ../../../opencv/cppbuild/$PLATFORM/include/opencv/cvaux.h include
cp -r ../../../opencv/cppbuild/$PLATFORM/include/opencv2/ include

mkdir -p bin
cp -r ../flandmark/build/x64/libflandmark/RelWithDebInfo/flandmark_shared.dll bin
cp -r ../../../opencv/cppbuild/$PLATFORM/bin/* bin

mkdir -p lib
cp -r ../flandmark/build/x64/libflandmark/RelWithDebInfo/flandmark_static.lib lib
cp -r ../../../opencv/cppbuild/$PLATFORM/lib/* lib

cd ..