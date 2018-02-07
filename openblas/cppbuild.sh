#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" openblas
    popd
    exit
fi

OPENBLAS_VERSION=0.2.20

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
        patch -Np1 < ../../../OpenBLAS-android.patch
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
        export TARGET=ARMV5 # to disable hard-float functions unsupported by Android
        export ARM_SOFTFP_ABI=1
        sed -i 's/-march=armv5/-march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16/' Makefile.arm
        ;;
    android-x86)
        patch -Np1 < ../../../OpenBLAS-android.patch
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
    ios-arm)
        export CC="$(xcrun --sdk iphoneos --find clang) -isysroot $(xcrun --sdk iphoneos --show-sdk-path) -arch armv7 -miphoneos-version-min=5.0"
        export FC=
        export NO_LAPACK=1
        export NOFORTRAN=1
        export BINARY=32
        export TARGET=ARMV5 # to disable unsupported assembler from iOS SDK
        export NO_SHARED=1
        ;;
    ios-arm64)
        # https://gmplib.org/list-archives/gmp-bugs/2014-September/003538.html
        sed -i="" 's/add.sp, sp, #-(11 \* 16)/sub sp, sp, #(11 \* 16)/g' kernel/arm64/sgemm_kernel_4x4.S
        # but still results in a linking error, so disable the assembler entirely
        sed -i="" 's/sgemm_kernel_4x4.S/..\/generic\/gemmkernel_2x2.c/g' kernel/arm64/KERNEL.ARMV8
        export CC="$(xcrun --sdk iphoneos --find clang) -isysroot $(xcrun --sdk iphoneos --show-sdk-path) -arch arm64 -miphoneos-version-min=5.0"
        export FC=
        export NO_LAPACK=1
        export NOFORTRAN=1
        export BINARY=64
        export TARGET=ARMV8
        export NO_SHARED=1
        ;;
    ios-x86)
        export CC="$(xcrun --sdk iphonesimulator --find clang) -isysroot $(xcrun --sdk iphonesimulator --show-sdk-path) -arch i686 -mios-simulator-version-min=5.0"
        export FC=
        export NO_LAPACK=1
        export NOFORTRAN=1
        export BINARY=32
        export TARGET=ATOM
        export NO_SHARED=1
        ;;
    ios-x86_64)
        export CC="$(xcrun --sdk iphonesimulator --find clang) -isysroot $(xcrun --sdk iphonesimulator --show-sdk-path) -arch x86_64 -mios-simulator-version-min=5.0"
        export FC=
        export NO_LAPACK=1
        export NOFORTRAN=1
        export BINARY=64
        export TARGET=ATOM
        export NO_SHARED=1
        ;;
    linux-x86)
        export CC="$OLDCC -m32"
        export FC="$OLDFC -m32"
        export BINARY=32
        export DYNAMIC_ARCH=1
        export TARGET=CORE2
        ;;
    linux-x86_64)
        export CC="$OLDCC -m64"
        export FC="$OLDFC -m64"
        export BINARY=64
        export DYNAMIC_ARCH=1
        export TARGET=HASWELL
        export NO_AVX2=1
        ;;
    linux-ppc64le)
        # patch to use less buggy generic kernels
        patch -Np1 < ../../../OpenBLAS-linux-ppc64le.patch
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          export CC="$OLDCC -m64"
          export FC="$OLDFC -m64"
        else
          export CC="powerpc64le-linux-gnu-gcc"
          export FC="powerpc64le-linux-gnu-gfortran"
          export CROSS_SUFFIX="powerpc64le-linux-gnu-"
        fi
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
        patch -Np1 < ../../../OpenBLAS-macosx.patch
        export CC="$(ls -1 /usr/local/bin/gcc-? | head -n 1)"
        export FC="$(ls -1 /usr/local/bin/gfortran-? | head -n 1)"
        export BINARY=64
        export DYNAMIC_ARCH=1
        export LDFLAGS="-static-libgcc -static-libgfortran -lgfortran /usr/local/lib/gcc/?/libquadmath.a"
        export TARGET=HASWELL
        export NO_AVX2=1
        ;;
    windows-x86)
        export CC="$OLDCC -m32"
        export FC="$OLDFC -m32"
        export BINARY=32
        export DYNAMIC_ARCH=1
        export LDFLAGS="-static-libgcc -static-libgfortran -Wl,-Bstatic -lgfortran -lgcc -lgcc_eh -lpthread"
        export TARGET=CORE2
        ;;
    windows-x86_64)
        export CC="$OLDCC -m64"
        export FC="$OLDFC -m64"
        export BINARY=64
        export DYNAMIC_ARCH=1
        export LDFLAGS="-static-libgcc -static-libgfortran -Wl,-Bstatic -lgfortran -lgcc -lgcc_eh -lpthread"
        export TARGET=HASWELL
        export NO_AVX2=1
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

make -s -j $MAKEJ libs netlib shared "CROSS_SUFFIX=$CROSS_SUFFIX" "CC=$CC" "FC=$FC" "HOSTCC=$HOSTCC" BINARY=$BINARY COMMON_PROF= F_COMPILER=GFORTRAN
make install "PREFIX=$INSTALL_PATH"
unset CC
unset LDFLAGS

cd ../..
