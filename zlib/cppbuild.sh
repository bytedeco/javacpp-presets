#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" zlib
    popd
    exit
fi

ZLIB_VERSION=1.2.11
download http://zlib.net/zlib-$ZLIB_VERSION.tar.gz zlib-$ZLIB_VERSION.tar.gz
mkdir -p $PLATFORM
cd $PLATFORM
tar -xzvf ../zlib-$ZLIB_VERSION.tar.gz
cd zlib-$ZLIB_VERSION

case $PLATFORM in
    android-arm)
        CC="$ANDROID_CC" CFLAGS="$ANDROID_FLAGS" ./configure --prefix=.. --static
        make -j $MAKEJ
        make install
        ;;
    android-arm64)
        CC="$ANDROID_CC" CFLAGS="$ANDROID_FLAGS" ./configure --prefix=.. --static
        make -j $MAKEJ
        make install
        ;;
    android-x86)
        CC="$ANDROID_CC" CFLAGS="$ANDROID_FLAGS" ./configure --prefix=.. --static
        make -j $MAKEJ
        make install
        ;;
    android-x86_64)
        CC="$ANDROID_CC" CFLAGS="$ANDROID_FLAGS" ./configure --prefix=.. --static
        make -j $MAKEJ
        make install
        ;;
    linux-x86)
        CC="gcc -m32 -fPIC" ./configure --prefix=.. --static
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        CC="gcc -m64 -fPIC" ./configure --prefix=.. --static
        make -j $MAKEJ
        make install
        ;;
    macosx-x86_64)
        ./configure --prefix=.. --static
        make -j $MAKEJ
        make install
        ;;
    windows-x86)
        nmake -f win32/Makefile.msc zlib.lib
        mkdir -p ../include ../lib
        cp zconf.h zlib.h ../include/
        cp zlib.lib ../lib/
        ;;
    windows-x86_64)
        nmake -f win32/Makefile.msc zlib.lib
        mkdir -p ../include ../lib
        cp zconf.h zlib.h ../include/
        cp zlib.lib ../lib/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
