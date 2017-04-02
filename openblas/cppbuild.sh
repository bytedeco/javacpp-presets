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

# blas (requires fortran, e.g. sudo yum install gcc-gfortran)
export CROSS_SUFFIX=
export HOSTCC=gcc
export NO_LAPACK=0
export NUM_THREADS=64
export NO_AFFINITY=1
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
            export NOFORTRAN=1
        fi
        export BINARY=32
        export TARGET=ARMV5
        sed -i 's/-march=armv5/-march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16/' Makefile.arm
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
            export NOFORTRAN=1
        fi
        export BINARY=32
        export TARGET=ATOM
        ;;
    linux-x86)
        export CC="$OLDCC -m32"
        export FC="$OLDFC -m32"
        export BINARY=32
        export DYNAMIC_ARCH=1
        ;;
    linux-x86_64)
        export CC="$OLDCC -m64"
        export FC="$OLDFC -m64"
        export BINARY=64
        export DYNAMIC_ARCH=1
        ;;
    linux-ppc64le)
        # patch to use less buggy generic kernels
        patch -Np1 < ../../../OpenBLAS-$OPENBLAS_VERSION-linux-ppc64le.patch
        export CC="$OLDCC -m64"
        export FC="$OLDFC -m64"
        export BINARY=64
        export TARGET=POWER5
        ;;
    linux-armhf)
        export CC="arm-linux-gnueabihf-gcc"
        export FC="arm-linux-gnueabihf-gfortran"
        export BINARY=32
        export TARGET=ARMV6
        ;;
    macosx-*)
        patch -Np1 < ../../../OpenBLAS-$OPENBLAS_VERSION-macosx.patch
        export CC="$(ls -1 /usr/local/bin/gcc-? | head -n 1)"
        export FC="$(ls -1 /usr/local/bin/gfortran-? | head -n 1)"
        export BINARY=64
        export DYNAMIC_ARCH=1
        export LDFLAGS="-static-libgcc -static-libgfortran -lgfortran /usr/local/opt/gcc?/lib/gcc/?/libquadmath.a"
        ;;
    windows-x86)
        export CC="$OLDCC -m32"
        export FC="$OLDFC -m32"
        export BINARY=32
        export DYNAMIC_ARCH=1
        export LDFLAGS="-static-libgcc -static-libgfortran -Wl,-Bstatic -lgfortran -lgcc -lgcc_eh -lpthread"
        ;;
    windows-x86_64)
        export CC="$OLDCC -m64"
        export FC="$OLDFC -m64"
        export BINARY=64
        export DYNAMIC_ARCH=1
        export LDFLAGS="-static-libgcc -static-libgfortran -Wl,-Bstatic -lgfortran -lgcc -lgcc_eh -lpthread"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

make -j $MAKEJ libs netlib shared "CROSS_SUFFIX=$CROSS_SUFFIX" "CC=$CC" "FC=$FC" "HOSTCC=$HOSTCC" BINARY=$BINARY COMMON_PROF= F_COMPILER=GFORTRAN
make install "PREFIX=$INSTALL_PATH"
export LDFLAGS=

cd ../..
