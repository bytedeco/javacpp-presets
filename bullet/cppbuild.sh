#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" bullet
    popd
    exit
fi

BULLET_VERSION=3.24
download https://github.com/bulletphysics/bullet3/archive/refs/tags/$BULLET_VERSION.zip bullet-$BULLET_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
unzip -qo ../bullet-$BULLET_VERSION.zip
cd bullet3-$BULLET_VERSION
mkdir -p build
cd build

case $PLATFORM in
    android-arm)
        cmake \
            -DANDROID_ABI=armeabi-v7a \
            -DANDROID_NATIVE_API_LEVEL=24 \
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
            -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake \
            -DENABLE_VHACD=OFF \
            -DUSE_DOUBLE_PRECISION=OFF \
            -DUSE_GLUT=OFF \
            -DUSE_GRAPHICAL_BENCHMARK=OFF \
            ..
        make -j $MAKEJ
        make install/strip
        ;;
    android-arm64)
        cmake \
            -DANDROID_ABI=arm64-v8a \
            -DANDROID_NATIVE_API_LEVEL=24 \
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
            -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake \
            -DENABLE_VHACD=OFF \
            -DUSE_DOUBLE_PRECISION=OFF \
            -DUSE_GLUT=OFF \
            -DUSE_GRAPHICAL_BENCHMARK=OFF \
            ..
        make -j $MAKEJ
        make install/strip
        ;;
    android-x86)
        cmake \
            -DANDROID_ABI=x86 \
            -DANDROID_NATIVE_API_LEVEL=24 \
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
            -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake \
            -DENABLE_VHACD=OFF \
            -DUSE_DOUBLE_PRECISION=OFF \
            -DUSE_GLUT=OFF \
            -DUSE_GRAPHICAL_BENCHMARK=OFF \
            ..
        make -j $MAKEJ
        make install/strip
        ;;
    android-x86_64)
        cmake \
            -DANDROID_ABI=x86_64 \
            -DANDROID_NATIVE_API_LEVEL=24 \
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
            -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake \
            -DENABLE_VHACD=OFF \
            -DUSE_DOUBLE_PRECISION=OFF \
            -DUSE_GLUT=OFF \
            -DUSE_GRAPHICAL_BENCHMARK=OFF \
            ..
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
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
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
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
        ;;
    macosx-x86_64)
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
            -DCMAKE_MACOSX_RPATH=ON \
            ..
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86|windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"
        cmake \
            -G "Ninja" \
            -DBUILD_BULLET2_DEMOS=OFF \
            -DBUILD_CLSOCKET=OFF \
            -DBUILD_CPU_DEMOS=OFF \
            -DBUILD_EGL=OFF \
            -DBUILD_ENET=OFF \
            -DBUILD_EXTRAS=OFF \
            -DBUILD_OPENGL3_DEMOS=OFF \
            -DBUILD_SHARED_LIBS=OFF \
            -DBUILD_UNIT_TESTS=OFF \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
            -DENABLE_VHACD=OFF \
            -DINSTALL_LIBS=ON \
            -DUSE_DOUBLE_PRECISION=OFF \
            -DUSE_GLUT=OFF \
            -DUSE_GRAPHICAL_BENCHMARK=OFF \
            -DUSE_MSVC_RUNTIME_LIBRARY_DLL=ON \
            ..
        ninja -j $MAKEJ
        ninja install
	;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

sedinplace "s/\(typedef.*btSolverCallback.*;\)/public: \1 private:/g" \
    ${INSTALL_PATH}/include/bullet/BulletSoftBody/btDeformableMultiBodyDynamicsWorld.h

cd ../../..
