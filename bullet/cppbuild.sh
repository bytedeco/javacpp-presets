#!/bin/bash

# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" opencl
    popd
    exit
fi

BULLET_VERSION=3.21
download https://github.com/bulletphysics/bullet3/archive/refs/tags/$BULLET_VERSION.zip bullet-$BULLET_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
unzip -q ../bullet-$BULLET_VERSION.zip

case $PLATFORM in
    linux-x86_64)
        cd bullet3-$BULLET_VERSION
        mkdir .build
        cd .build
        cmake \
            -DBUILD_BULLET2_DEMOS=OFF \
            -DBUILD_CLSOCKET=OFF \
            -DBUILD_CPU_DEMOS=OFF \
            -DBUILD_EGL=OFF \
            -DBUILD_ENET=OFF \
            -DBUILD_EXTRAS=OFF \
            -DBUILD_OPENGL3_DEMOS=OFF \
            -DBUILD_SHARED_LIBS=ON \
            -DBUILD_UNIT_TESTS=OFF \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
            -DENABLE_VHACD=OFF \
            -DUSE_DOUBLE_PRECISION=OFF \
            -DUSE_GLUT=OFF \
            -DUSE_GRAPHICAL_BENCHMARK=OFF \
            ..
        make -j $MAKEJ
        make install/strip
        cd ../..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ..
