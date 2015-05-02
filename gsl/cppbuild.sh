#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" gsl
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    GSL_VERSION=1.16-2
    [[ $PLATFORM == *64 ]] && BITS=64 || BITS=32
    download http://mirrors.kernel.org/fedora/releases/21/Everything/x86_64/os/Packages/m/mingw$BITS-gsl-$GSL_VERSION.fc21.noarch.rpm mingw$BITS-gsl-$GSL_VERSION.rpm

    mkdir -p $PLATFORM
    cd $PLATFORM
    rm -Rf include lib bin
    /C/Program\ Files/7-Zip/7z x -y ../mingw$BITS-gsl-$GSL_VERSION.rpm -o..
    /C/Program\ Files/7-Zip/7z x -y ../mingw$BITS-gsl-$GSL_VERSION.cpio
else
    GSL_VERSION=1.16
    download ftp://ftp.gnu.org/gnu/gsl/gsl-$GSL_VERSION.tar.gz gsl-$GSL_VERSION.tar.gz

    mkdir -p $PLATFORM
    cd $PLATFORM
    INSTALL_PATH=`pwd`
    tar -xzvf ../gsl-$GSL_VERSION.tar.gz
    cd gsl-$GSL_VERSION
fi

case $PLATFORM in
    android-arm)
        ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ;;
     android-x86)
        ./configure --prefix=$INSTALL_PATH --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ;;
    linux-x86)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m32"
        make -j4
        make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m64"
        make -j4
        make install-strip
        ;;
    macosx-*)
        patch -Np1 < ../../../gsl-$GSL_VERSION-macosx.patch
        ./configure --prefix=$INSTALL_PATH
        make -j4
        make install-strip
        ;;
    windows-x86)
        mv usr/i686-w64-mingw32/sys-root/mingw/* .
        ;;
    windows-x86_64)
        mv usr/x86_64-w64-mingw32/sys-root/mingw/* .
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
