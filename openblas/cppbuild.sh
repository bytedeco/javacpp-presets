#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" openblas
    popd
    exit
fi

OPENBLAS_VERSION=0.3.19

download https://github.com/xianyi/OpenBLAS/archive/v$OPENBLAS_VERSION.tar.gz OpenBLAS-$OPENBLAS_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib bin OpenBLAS-$OPENBLAS_VERSION-nolapack
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xzf ../OpenBLAS-$OPENBLAS_VERSION.tar.gz
tar --totals -xzf ../OpenBLAS-$OPENBLAS_VERSION.tar.gz --strip-components=1 -C OpenBLAS-$OPENBLAS_VERSION-nolapack/

cd OpenBLAS-$OPENBLAS_VERSION
cp lapack-netlib/LAPACKE/include/*.h ../include

# remove broken cross-compiler workaround on Mac
sedinplace '/if (($os eq "Darwin")/,/}/d' c_check ../OpenBLAS-$OPENBLAS_VERSION-nolapack/c_check
sedinplace 's/common.h/param.h/g' getarch_2nd.c ../OpenBLAS-$OPENBLAS_VERSION-nolapack/getarch_2nd.c

# blas (requires fortran, e.g. sudo yum install gcc-gfortran)
export FEXTRALIB="-lgfortran"
export CROSS_SUFFIX=
export HOSTCC=gcc
export NO_LAPACK=0
export NUM_THREADS=64
export NO_AFFINITY=1
export NO_AVX512=1
case $PLATFORM in
    android-arm)
        patch -Np1 < ../../../OpenBLAS-android.patch
        patch -Np1 -d ../OpenBLAS-$OPENBLAS_VERSION-nolapack/ < ../../../OpenBLAS-android.patch
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export FC="$ANDROID_PREFIX-gfortran $ANDROID_FLAGS"
        export CROSS_SUFFIX="$ANDROID_PREFIX-"
        export LDFLAGS="-ldl -lm -lc"
        if [[ ! -x "$ANDROID_PREFIX-gfortran" ]]; then
            export NO_LAPACK=1
            export NOFORTRAN=1
        fi
        export BINARY=32
        export TARGET=ARMV5 # to disable hard-float functions unsupported by Android
        export ARM_SOFTFP_ABI=1
        sedinplace 's/-march=armv5/-march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16/' Makefile.arm ../OpenBLAS-$OPENBLAS_VERSION-nolapack/Makefile.arm
        ;;
    android-arm64)
        patch -Np1 < ../../../OpenBLAS-android.patch
        patch -Np1 -d ../OpenBLAS-$OPENBLAS_VERSION-nolapack/ < ../../../OpenBLAS-android.patch
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export FC="$ANDROID_PREFIX-gfortran $ANDROID_FLAGS"
        export CROSS_SUFFIX="$ANDROID_PREFIX-"
        export LDFLAGS="-ldl -lm -lc"
        if [[ ! -x "$ANDROID_PREFIX-gfortran" ]]; then
            export NO_LAPACK=1
            export NOFORTRAN=1
        fi
        export BINARY=64
        export TARGET=ARMV8
        ;;
    android-x86)
        patch -Np1 < ../../../OpenBLAS-android.patch
        patch -Np1 -d ../OpenBLAS-$OPENBLAS_VERSION-nolapack/ < ../../../OpenBLAS-android.patch
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export FC="$ANDROID_PREFIX-gfortran $ANDROID_FLAGS"
        export CROSS_SUFFIX="$ANDROID_PREFIX-"
        export LDFLAGS="-ldl -lm -lc"
        if [[ ! -x "$ANDROID_PREFIX-gfortran" ]]; then
            export NO_LAPACK=1
            export NOFORTRAN=1
        fi
        export BINARY=32
        export TARGET=ATOM
        ;;
    android-x86_64)
        patch -Np1 < ../../../OpenBLAS-android.patch
        patch -Np1 -d ../OpenBLAS-$OPENBLAS_VERSION-nolapack/ < ../../../OpenBLAS-android.patch
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export FC="$ANDROID_PREFIX-gfortran $ANDROID_FLAGS"
        export CROSS_SUFFIX="$ANDROID_PREFIX-"
        export LDFLAGS="-ldl -lm -lc"
        if [[ ! -x "$ANDROID_PREFIX-gfortran" ]]; then
            export NO_LAPACK=1
            export NOFORTRAN=1
        fi
        export BINARY=64
        export TARGET=ATOM
        ;;
    ios-arm)
        patch -Np1 < ../../../OpenBLAS-ios.patch
        patch -Np1 -d ../OpenBLAS-$OPENBLAS_VERSION-nolapack/ < ../../../OpenBLAS-ios.patch
        export CC="$(xcrun --sdk iphoneos --find clang) -isysroot $(xcrun --sdk iphoneos --show-sdk-path) -arch armv7 -miphoneos-version-min=5.0"
        export FC=
        export NO_LAPACK=1
        export NOFORTRAN=1
        export BINARY=32
        export TARGET=ARMV5 # to disable unsupported assembler from iOS SDK: use Accelerate to optimize
        export NO_SHARED=1
        ;;
    ios-arm64)
        patch -Np1 < ../../../OpenBLAS-ios.patch
        patch -Np1 -d ../OpenBLAS-$OPENBLAS_VERSION-nolapack/ < ../../../OpenBLAS-ios.patch
        # use generic kernels as Xcode assembler does not accept optimized ones: use Accelerate to optimize
        cp kernel/arm/KERNEL.ARMV5 kernel/arm64/KERNEL.ARMV8
        cp kernel/arm/KERNEL.ARMV5 ../OpenBLAS-$OPENBLAS_VERSION-nolapack/kernel/arm64/KERNEL.ARMV8
        export CC="$(xcrun --sdk iphoneos --find clang) -isysroot $(xcrun --sdk iphoneos --show-sdk-path) -arch arm64 -miphoneos-version-min=5.0"
        export FC=
        export NO_LAPACK=1
        export NOFORTRAN=1
        export BINARY=64
        export TARGET=ARMV8
        export NO_SHARED=1
        ;;
    ios-x86)
        patch -Np1 < ../../../OpenBLAS-ios.patch
        patch -Np1 -d ../OpenBLAS-$OPENBLAS_VERSION-nolapack/ < ../../../OpenBLAS-ios.patch
        export CC="$(xcrun --sdk iphonesimulator --find clang) -isysroot $(xcrun --sdk iphonesimulator --show-sdk-path) -arch i686 -mios-simulator-version-min=5.0"
        export FC=
        export NO_LAPACK=1
        export NOFORTRAN=1
        export BINARY=32
        export TARGET=GENERIC # optimized kernels do not return correct results on iOS: use Accelerate to optimize
        export NO_SHARED=1
        ;;
    ios-x86_64)
        patch -Np1 < ../../../OpenBLAS-ios.patch
        patch -Np1 -d ../OpenBLAS-$OPENBLAS_VERSION-nolapack/ < ../../../OpenBLAS-ios.patch
        export CC="$(xcrun --sdk iphonesimulator --find clang) -isysroot $(xcrun --sdk iphonesimulator --show-sdk-path) -arch x86_64 -mios-simulator-version-min=5.0"
        export FC=
        export NO_LAPACK=1
        export NOFORTRAN=1
        export BINARY=64
        export TARGET=GENERIC # optimized kernels do not return correct results on iOS: use Accelerate to optimize
        export NO_SHARED=1
        ;;
    linux-x86)
        export CC="gcc -m32"
        export FC="gfortran -m32"
        export LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/'
        export BINARY=32
        export DYNAMIC_ARCH=1
        export TARGET=NORTHWOOD
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export FC="gfortran -m64"
        export LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/'
        export BINARY=64
        export DYNAMIC_ARCH=1
        export TARGET=NEHALEM
        export NO_AVX512=0
        ;;
    linux-ppc64le)
        # patch to use less buggy generic kernels
        patch -Np1 < ../../../OpenBLAS-linux-ppc64le.patch
        patch -Np1 -d ../OpenBLAS-$OPENBLAS_VERSION-nolapack/ < ../../../OpenBLAS-linux-ppc64le.patch
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          export CC="gcc -m64"
          export FC="gfortran -m64"
        else
          export CC="powerpc64le-linux-gnu-gcc"
          export FC="powerpc64le-linux-gnu-gfortran"
          export CROSS_SUFFIX="powerpc64le-linux-gnu-"
        fi
        export LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/'
        export BINARY=64
        export TARGET=POWER5
        ;;
    linux-mips64el)
        export CC="gcc -mabi=64"
        export FC="gfortran -mabi=64"
        export LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/ -Wl,-z,noexecstack'
        export BINARY=64
        export TARGET=MIPS
        ;;
    linux-armhf)
        export CC="arm-linux-gnueabihf-gcc"
        export FC="arm-linux-gnueabihf-gfortran"
        export LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/ -Wl,-z,noexecstack'
        export BINARY=32
        export TARGET=ARMV6
        ;;
    linux-arm64)
        export CC="aarch64-linux-gnu-gcc -mabi=lp64"
        export FC="aarch64-linux-gnu-gfortran"
        export LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/ -Wl,-z,noexecstack'
        export BINARY=64
        export TARGET=ARMV8
        ;;
    macosx-arm64)
        patch -Np1 < ../../../OpenBLAS-macosx.patch
        patch -Np1 -d ../OpenBLAS-$OPENBLAS_VERSION-nolapack/ < ../../../OpenBLAS-macosx.patch
        export CC="clang -arch arm64"
        export FC=
        export LDFLAGS='-s -Wl,-rpath,@loader_path/'
        export NO_LAPACK=1
        export NOFORTRAN=1
        export BINARY=64
        export TARGET=ARMV8
        ;;
    macosx-x86_64)
        patch -Np1 < ../../../OpenBLAS-macosx.patch
        patch -Np1 -d ../OpenBLAS-$OPENBLAS_VERSION-nolapack/ < ../../../OpenBLAS-macosx.patch
        export CC="$(ls -1 /usr/local/bin/gcc-? | head -n 1)"
        export FC="$(ls -1 /usr/local/bin/gfortran-? | head -n 1)"
        export LDFLAGS='-s -Wl,-rpath,@loader_path/ -lgfortran'
        export BINARY=64
        export DYNAMIC_ARCH=1
        export NO_AVX512=1
        export TARGET=NEHALEM
        ;;
    windows-x86)
        export CC="gcc -m32"
        export FC="gfortran -m32"
        export FEXTRALIB="-lgfortran -lquadmath"
        export BINARY=32
        export DYNAMIC_ARCH=1
        export LDFLAGS="-static-libgcc -static-libgfortran -Wl,-Bstatic -lgfortran -lquadmath -lgcc -lgcc_eh -lpthread"
        export TARGET=NORTHWOOD
        ;;
    windows-x86_64)
        export CC="gcc -m64"
        export FC="gfortran -m64"
        export FEXTRALIB="-lgfortran -lquadmath"
        export BINARY=64
        export DYNAMIC_ARCH=1
        export LDFLAGS="-static-libgcc -static-libgfortran -Wl,-Bstatic -lgfortran -lquadmath -lgcc -lgcc_eh -lpthread"
        export NO_AVX512=1
        export TARGET=NEHALEM
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

make -s -j $MAKEJ libs netlib shared "CROSS_SUFFIX=$CROSS_SUFFIX" "CC=$CC" "FC=$FC" "HOSTCC=$HOSTCC" BINARY=$BINARY COMMON_PROF= F_COMPILER=GFORTRAN "FEXTRALIB=$FEXTRALIB" USE_OPENMP=0 NUM_THREADS=$NUM_THREADS NO_AVX512=$NO_AVX512
make install "PREFIX=$INSTALL_PATH"

unset DYNAMIC_ARCH
cd ../OpenBLAS-$OPENBLAS_VERSION-nolapack/
make -s -j $MAKEJ libs netlib shared "CROSS_SUFFIX=$CROSS_SUFFIX" "CC=$CC" "FC=$FC" "HOSTCC=$HOSTCC" BINARY=$BINARY COMMON_PROF= F_COMPILER=GFORTRAN "FEXTRALIB=$FEXTRALIB" USE_OPENMP=0 NUM_THREADS=$NUM_THREADS NO_AVX512=$NO_AVX512 NO_LAPACK=1 LIBNAMESUFFIX=nolapack
make install "PREFIX=$INSTALL_PATH" NO_LAPACK=1 LIBNAMESUFFIX=nolapack

unset CC
unset FC
unset LDFLAGS

if [[ -f ../lib/libopenblas.dll.a ]]; then
    # bundle the import library for Windows under a friendly name for MSVC
    cp ../lib/libopenblas.dll.a ../lib/openblas.lib
    cp ../lib/libopenblas_nolapack.dll.a ../lib/openblas_nolapack.lib
fi

cd ../..
