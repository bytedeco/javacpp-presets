#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" zlib
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    ZLIB_VERSION=128
    download http://zlib.net/zlib$ZLIB_VERSION-dll.zip zlib$ZLIB_VERSION-dll.zip
    mkdir -p $PLATFORM
    cd $PLATFORM
    unzip ../zlib$ZLIB_VERSION-dll.zip -d zlib$ZLIB_VERSION-dll
    cd zlib$ZLIB_VERSION-dll
else
    ZLIB_VERSION=1.2.11
    download http://zlib.net/zlib-$ZLIB_VERSION.tar.gz zlib-$ZLIB_VERSION.tar.gz
    mkdir -p $PLATFORM
    cd $PLATFORM
    tar -xzvf ../zlib-$ZLIB_VERSION.tar.gz
    cd zlib-$ZLIB_VERSION
fi

case $PLATFORM in
    android-arm)
        CC="$ANDROID_BIN-gcc" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc" ./configure --prefix=.. --static
        make -j4
        make install
        ;;
    android-x86)
        CC="$ANDROID_BIN-gcc" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc" ./configure --prefix=.. --static
        make -j4
        make install
        ;;
    linux-x86)
        CC="gcc -m32 -fPIC" ./configure --prefix=.. --static
        make -j4
        make install
        ;;
    linux-x86_64)
        CC="gcc -m64 -fPIC" ./configure --prefix=.. --static
        make -j4
        make install
        ;;
    macosx-x86_64)
        ./configure --prefix=.. --static
        make -j4
        make install
        ;;
    windows-x86)
        cp -r include ..
        cp -r lib ..
        mkdir -p ../bin
        cp *.dll ../bin
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..