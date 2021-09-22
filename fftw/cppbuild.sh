#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" fftw
    popd
    exit
fi

FFTW_VERSION=3.3.10
download http://www.fftw.org/fftw-$FFTW_VERSION.tar.gz fftw-$FFTW_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../fftw-$FFTW_VERSION.tar.gz
cd fftw-$FFTW_VERSION

case $PLATFORM in
    android-arm)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="-ldl -lm -lc"
        patch -Np1 < ../../../fftw-android.patch
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
    android-arm64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="-ldl -lm -lc"
        patch -Np1 < ../../../fftw-android.patch
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host="aarch64-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host="aarch64-linux-android" --with-sysroot="$ANDROID_ROOT" --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
     android-x86)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="-ldl -lm -lc"
        patch -Np1 < ../../../fftw-android.patch
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
     android-x86_64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="-ldl -lm -lc"
        patch -Np1 < ../../../fftw-android.patch
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="x86_64-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="x86_64-linux-android" --with-sysroot="$ANDROID_ROOT" --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
    linux-x86)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m32"
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m32" --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m64"
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m64" --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
    linux-armhf)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host=arm-linux-gnueabihf
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host=arm-linux-gnueabihf --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
    linux-arm64)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host=aarch64-linux-gnu
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host=aarch64-linux-gnu --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads CC="gcc -m64"
          make -j $MAKEJ V=0
          make install-strip
          ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads CC="gcc -m64" --enable-float
          make -j $MAKEJ V=0
          make install-strip
        else
          CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host=powerpc64le-linux-gnu --build=ppc64le-linux
          make -j $MAKEJ V=0
          make install-strip
          CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host=powerpc64le-linux-gnu --build=ppc64le-linux --enable-float
          make -j $MAKEJ V=0
          make install-strip
        fi
        ;;
    linux-mips64el)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads CC="gcc -mabi=64"
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads CC="gcc -mabi=64" --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
    macosx-*)
        patch -Np1 < ../../../fftw-macosx.patch
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
    windows-x86)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m32" --with-our-malloc
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m32" --with-our-malloc --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
    windows-x86_64)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m64" --with-our-malloc
        make -j $MAKEJ V=0
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m64" --with-our-malloc --enable-float
        make -j $MAKEJ V=0
        make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
