#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" leptonica
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    LEPTONICA_VERSION=1.71-1
    [[ $PLATFORM == *64 ]] && BITS=64 || BITS=32
    download http://mirrors.kernel.org/fedora/releases/21/Everything/x86_64/os/Packages/m/mingw$BITS-leptonica-$LEPTONICA_VERSION.fc21.noarch.rpm mingw$BITS-leptonica-$LEPTONICA_VERSION.fc21.noarch.rpm
    download http://mirrors.kernel.org/fedora/releases/21/Everything/x86_64/os/Packages/m/mingw$BITS-giflib-5.0.5-2.fc21.noarch.rpm mingw$BITS-giflib-5.0.5-2.fc21.noarch.rpm
    download http://mirrors.kernel.org/fedora/releases/21/Everything/x86_64/os/Packages/m/mingw$BITS-libjpeg-turbo-1.3.1-3.fc21.noarch.rpm mingw$BITS-libjpeg-turbo-1.3.1-3.fc21.noarch.rpm
    download http://mirrors.kernel.org/fedora/releases/21/Everything/x86_64/os/Packages/m/mingw$BITS-libpng-1.6.10-2.fc21.noarch.rpm mingw$BITS-libpng-1.6.10-2.fc21.noarch.rpm
    download http://mirrors.kernel.org/fedora/releases/21/Everything/x86_64/os/Packages/m/mingw$BITS-libtiff-4.0.3-5.fc21.noarch.rpm mingw$BITS-libtiff-4.0.3-5.fc21.noarch.rpm
    download http://mirrors.kernel.org/fedora/releases/21/Everything/x86_64/os/Packages/m/mingw$BITS-libwebp-0.4.2-1.fc21.noarch.rpm mingw$BITS-libwebp-0.4.2-1.fc21.noarch.rpm
    download http://mirrors.kernel.org/fedora/releases/21/Everything/x86_64/os/Packages/m/mingw$BITS-zlib-1.2.8-3.fc21.noarch.rpm mingw$BITS-zlib-1.2.8-3.fc21.noarch.rpm

    function extract {
        /C/Program\ Files/7-Zip/7z x -y $1
    }
    extract mingw$BITS-leptonica-*.rpm
    extract mingw$BITS-giflib-*.rpm
    extract mingw$BITS-libjpeg-turbo-*.rpm
    extract mingw$BITS-libpng-*.rpm
    extract mingw$BITS-libtiff-*.rpm
    extract mingw$BITS-libwebp-*.rpm
    extract mingw$BITS-zlib-*.rpm

    mkdir -p $PLATFORM
    cd $PLATFORM
    rm -Rf include lib bin
    extract ../mingw$BITS-leptonica-*.cpio
    extract ../mingw$BITS-giflib-*.cpio
    extract ../mingw$BITS-libjpeg-turbo-*.cpio
    extract ../mingw$BITS-libpng-*.cpio
    extract ../mingw$BITS-libtiff-*.cpio
    extract ../mingw$BITS-libwebp-*.cpio
    extract ../mingw$BITS-zlib-*.cpio
else
    LEPTONICA_VERSION=1.71
    download http://www.leptonica.org/source/leptonica-$LEPTONICA_VERSION.tar.gz leptonica-$LEPTONICA_VERSION.tar.gz

    mkdir -p $PLATFORM
    cd $PLATFORM
    INSTALL_PATH=`pwd`
    tar -xzvf ../leptonica-$LEPTONICA_VERSION.tar.gz
    cd leptonica-$LEPTONICA_VERSION
fi

case $PLATFORM in
    android-arm)
        patch -Np1 < ../../../leptonica-$LEPTONICA_VERSION-android.patch
        ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ;;
     android-x86)
        patch -Np1 < ../../../leptonica-$LEPTONICA_VERSION-android.patch
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
