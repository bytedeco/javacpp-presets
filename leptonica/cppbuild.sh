if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

if [[ $PLATFORM == windows* ]]; then
    LEPTONICA_VERSION=1.68
    [[ $PLATFORM == *64 ]] && BITS=64 || BITS=32
    download http://www.leptonica.org/source/leptonica-$LEPTONICA_VERSION-win$BITS-lib-include-dirs.zip leptonica-$LEPTONICA_VERSION-win$BITS-lib-include-dirs.zip

    mkdir -p $PLATFORM
    cd $PLATFORM
    unzip -o ../leptonica-$LEPTONICA_VERSION-win$BITS-lib-include-dirs.zip
    cd include
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
        ./configure --prefix=$INSTALL_PATH
        make -j4
        make install-strip
        ;;
    windows-x86)
        ;;
    windows-x86_64)
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
