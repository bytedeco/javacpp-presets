#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" opencascade
    popd
    exit
fi

OPENCASCADE_VERSION=7.4.0
GITHUB_RELEASE=https://github.com/tpaviot/oce/releases/download/official-upstream-packages/opencascade-$OPENCASCADE_VERSION.tgz
download $GITHUB_RELEASE opencascade-$OPENCASCADE_VERSION.tgz

mkdir -p $PLATFORM
cd $PLATFORM
tar -xzvf ../opencascade-$OPENCASCADE_VERSION.tgz
cd opencascade-$OPENCASCADE_VERSION

case $PLATFORM in
    android-arm)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=android-arm.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    android-arm64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=android-arm64.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=android-x86.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        ;;
    android-x86_64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=android-x86_64.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. .
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
