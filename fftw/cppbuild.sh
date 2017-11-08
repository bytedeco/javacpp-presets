#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" fftw
    popd
    exit
fi

FFTW_VERSION=3.3.7
download http://www.fftw.org/fftw-$FFTW_VERSION.tar.gz fftw-$FFTW_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../fftw-$FFTW_VERSION.tar.gz
cd fftw-$FFTW_VERSION

case $PLATFORM in
    android-arm)
        export AR="$ANDROID_BIN-ar"
        export RANLIB="$ANDROID_BIN-ranlib"
        export CPP="$ANDROID_BIN-cpp"
        export CC="$ANDROID_BIN-gcc"
        export STRIP="$ANDROID_BIN-strip"
        export CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID"
        export CFLAGS="$CPPFLAGS -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        export LDFLAGS="-nostdlib -Wl,--fix-cortex-a8 -z text"
        export LIBS="-lgcc -ldl -lz -lm -lc"
        patch -Np1 < ../../../fftw-android.patch
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
     android-x86)
        export AR="$ANDROID_BIN-ar"
        export RANLIB="$ANDROID_BIN-ranlib"
        export CPP="$ANDROID_BIN-cpp"
        export CC="$ANDROID_BIN-gcc"
        export STRIP="$ANDROID_BIN-strip"
        export CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID"
        export CFLAGS="$CPPFLAGS -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        export LDFLAGS="-nostdlib -z text"
        export LIBS="-lgcc -ldl -lz -lm -lc"
        patch -Np1 < ../../../fftw-android.patch
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="$OLDCC -m32"
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="$OLDCC -m32" --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="$OLDCC -m64"
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="$OLDCC -m64" --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    linux-armhf)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host=arm-linux-gnueabihf --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads CC="$OLDCC -m64"
          make -j $MAKEJ
          make install-strip
          ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads CC="$OLDCC -m64" --enable-float
          make -j $MAKEJ
          make install-strip
        else
          CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host=powerpc64le-linux-gnu --build=ppc64le-linux
          make -j $MAKEJ
          make install-strip
          CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --host=powerpc64le-linux-gnu --build=ppc64le-linux --enable-float
          make -j $MAKEJ
          make install-strip
        fi
        ;;
    macosx-*)
        patch -Np1 < ../../../fftw-macosx.patch
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m32" --with-our-malloc
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m32" --with-our-malloc --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86_64)
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m64" --with-our-malloc
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-fortran --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m64" --with-our-malloc --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
