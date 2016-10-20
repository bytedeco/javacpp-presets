#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" openblas
    popd
    exit
fi

OPENBLAS_VERSION=0.2.19

download https://github.com/xianyi/OpenBLAS/archive/v$OPENBLAS_VERSION.tar.gz OpenBLAS-$OPENBLAS_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xzf ../OpenBLAS-$OPENBLAS_VERSION.tar.gz

cd OpenBLAS-$OPENBLAS_VERSION

export CROSS_SUFFIX=
export HOSTCC=gcc
export NO_LAPACK=0
export TARGET=GENERIC
case $PLATFORM in
    android-arm)
        patch -Np1 < ../../../OpenBLAS-$OPENBLAS_VERSION-android.patch
        export CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        export CC="$ANDROID_BIN-gcc $CFLAGS"
        export FC="$ANDROID_BIN-gfortran $CFLAGS"
        export CROSS_SUFFIX="$ANDROID_BIN-"
        export LDFLAGS="-Wl,--fix-cortex-a8 -Wl,--no-undefined -z text -lgcc -ldl -lz -lm -lc"
        if [[ ! -x "$ANDROID_BIN-gfortran" ]]; then
            export NO_LAPACK=1
        fi
        export BINARY=32
        export TARGET=ARMV5
        ;;
    android-x86)
        patch -Np1 < ../../../OpenBLAS-$OPENBLAS_VERSION-android.patch
        export CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        export CC="$ANDROID_BIN-gcc $CFLAGS"
        export FC="$ANDROID_BIN-gfortran $CFLAGS"
        export CROSS_SUFFIX="$ANDROID_BIN-"
        export LDFLAGS="-Wl,--no-undefined -z text -lgcc -ldl -lz -lm -lc"
        if [[ ! -x "$ANDROID_BIN-gfortran" ]]; then
            export NO_LAPACK=1
        fi
        export BINARY=32
        export TARGET=ATOM
        ;;
    linux-x86)
        export CC="$OLDCC -m32"
        export FC="$OLDFC -m32"
        export BINARY=32
        ;;
    linux-x86_64)
        export CC="$OLDCC -m64"
        export FC="$OLDFC -m64"
        export BINARY=64
        ;;
    linux-ppc64le)
        export CC="$OLDCC -m64"
        export FC="$OLDFC -m64"
        export BINARY=64
        export TARGET=POWER8
        ;;
    linux-armhf)
        export CC="arm-linux-gnueabihf-gcc"
        export FC="arm-linux-gnueabihf-gfortran"
        export BINARY=32
        export TARGET=ARMV6
        ;;
    macosx-*)
        export CC="gcc"
        export FC="gfortran"
        export BINARY=64
        ;;
    windows-x86)
        export CC="$OLDCC -m32"
        export FC="$OLDFC -m32"
        export BINARY=32
        ;;
    windows-x86_64)
        export CC="$OLDCC -m64"
        export FC="$OLDFC -m64"
        export BINARY=64
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

make -j $MAKEJ "CROSS_SUFFIX=$CROSS_SUFFIX" "CC=$CC" "FC=$FC" "HOSTCC=$HOSTCC" BINARY=$BINARY TARGET=$TARGET COMMON_PROF=
make install "PREFIX=$INSTALL_PATH"

cd ../..
