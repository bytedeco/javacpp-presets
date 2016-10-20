#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" fftw
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    FFTW_VERSION=3.3.5
    [[ $PLATFORM == *64 ]] && BITS=64 || BITS=32
    download ftp://ftp.fftw.org/pub/fftw/fftw-$FFTW_VERSION-dll$BITS.zip fftw-$FFTW_VERSION-dll$BITS.zip

    mkdir -p $PLATFORM
    cd $PLATFORM
    mkdir -p include lib
    unzip -o ../fftw-$FFTW_VERSION-dll$BITS.zip -d fftw-$FFTW_VERSION-dll$BITS
    cd fftw-$FFTW_VERSION-dll$BITS
else
    FFTW_VERSION=3.3.5
    download http://www.fftw.org/fftw-$FFTW_VERSION.tar.gz fftw-$FFTW_VERSION.tar.gz

    mkdir -p $PLATFORM
    cd $PLATFORM
    INSTALL_PATH=`pwd`
    tar -xzvf ../fftw-$FFTW_VERSION.tar.gz
    cd fftw-$FFTW_VERSION
fi

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
        patch -Np1 < ../../../fftw-$FFTW_VERSION-android.patch
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" --enable-float
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
        patch -Np1 < ../../../fftw-$FFTW_VERSION-android.patch
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86)
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="$OLDCC -m32"
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="$OLDCC -m32" --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="$OLDCC -m64"
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="$OLDCC -m64" --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    linux-armhf)
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf 
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    linux-ppc64le)
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads CC="$OLDCC -m64"
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads CC="$OLDCC -m64" --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    macosx-*)
        patch -Np1 < ../../../fftw-$FFTW_VERSION-macosx.patch
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2
        make -j $MAKEJ
        make install-strip
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-float
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        cp *.h ../include
        cp *.dll ../lib
        ;;
    windows-x86_64)
        cp *.h ../include
        cp *.dll ../lib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
