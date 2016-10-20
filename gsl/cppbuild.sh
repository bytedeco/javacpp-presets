#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" gsl
    popd
    exit
fi

GSL_VERSION=2.2.1
download ftp://ftp.gnu.org/gnu/gsl/gsl-$GSL_VERSION.tar.gz gsl-$GSL_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
tar -xzvf ../gsl-$GSL_VERSION.tar.gz
cd gsl-$GSL_VERSION

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
        export GSL_LDFLAGS="-Lcblas/.libs/ -lgslcblas"
        patch -Np1 < ../../../gsl-$GSL_VERSION-android.patch
        ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT"
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
        export GSL_LDFLAGS="-Lcblas/.libs/ -lgslcblas"
        patch -Np1 < ../../../gsl-$GSL_VERSION-android.patch
        ./configure --prefix=$INSTALL_PATH --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86)
        ./configure --prefix=$INSTALL_PATH CC="$OLDCC -m32"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH CC="$OLDCC -m64"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-armhf)
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install-strip
        ;;
    linux-ppc64le)
        ./configure --prefix=$INSTALL_PATH CC="$OLDCC -m64"
        make -j $MAKEJ
        make install-strip
        ;;
    macosx-*)
        patch -Np1 < ../../../gsl-$GSL_VERSION-macosx.patch
        ./configure --prefix=$INSTALL_PATH
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m32 -static-libgcc"
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86_64)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m64 -static-libgcc"
        make -j $MAKEJ
        make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
