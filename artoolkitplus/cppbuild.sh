#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" artoolkitplus
    popd
    exit
fi

ARTOOLKITPLUS_VERSION=2.3.1
download https://launchpad.net/artoolkitplus/trunk/$ARTOOLKITPLUS_VERSION/+download/ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2 ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2
# [[ -f ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2 ]] || curl -LO https://launchpad.net/artoolkitplus/trunk/$ARTOOLKITPLUS_VERSION/+download/ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
echo "Decompressing archives..."
tar --totals -xjf ../ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2 --exclude="*/id-markers/*"
cd ARToolKitPlus-$ARTOOLKITPLUS_VERSION
patch --binary -Np1 < ../../../ARToolKitPlus.patch || true

case $PLATFORM in
    android-arm)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    android-arm64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    android-x86_64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    linux-armhf)
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    linux-arm64)
        CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        else
          CC=powerpc64le-linux-gnu-gcc CXX=powerpc64le-linux-gnu-g++ CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DCMAKE_SYSTEM_PROCESSOR=powerpc -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        fi
        make -j $MAKEJ
        make install
        ;;
    linux-mips64el)
        CC="gcc -mabi=64" CXX="g++ -mabi=64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    windows-x86)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        nmake
        nmake install
        ;;
    windows-x86_64)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        nmake
        nmake install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
