if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

if [[ $PLATFORM == windows* ]]; then
    FFTW_VERSION=3.3.4
    download ftp://ftp.fftw.org/pub/fftw/fftw-$FFTW_VERSION-dll32.zip fftw-$FFTW_VERSION-dll32.zip
    download ftp://ftp.fftw.org/pub/fftw/fftw-$FFTW_VERSION-dll64.zip fftw-$FFTW_VERSION-dll64.zip

    INSTALL_DIR=/C/MinGW/local
    mkdir -p $INSTALL_DIR/include $INSTALL_DIR/lib32 $INSTALL_DIR/lib64 $INSTALL_DIR/bin32 $INSTALL_DIR/bin64
else
    FFTW_VERSION=3.3.4
    download http://www.fftw.org/fftw-$FFTW_VERSION.tar.gz fftw-$FFTW_VERSION.tar.gz

    tar -xzvf fftw-$FFTW_VERSION.tar.gz
    mv fftw-$FFTW_VERSION fftw-$FFTW_VERSION-$PLATFORM
    cd fftw-$FFTW_VERSION-$PLATFORM
fi

case $PLATFORM in
    android-arm)
        ./configure --enable-shared --enable-threads --with-combined-threads --host="arm-linux-androideabi" --prefix="$ANDROID_NDK/../local/" --libdir="$ANDROID_NDK/../local/lib/armeabi/" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ./configure --enable-shared --enable-threads --with-combined-threads --host="arm-linux-androideabi" --prefix="$ANDROID_NDK/../local/" --libdir="$ANDROID_NDK/../local/lib/armeabi/" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc" --enable-float
        make -j4
        make install-strip
        ;;
     android-x86)
        ./configure --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="i686-linux-android" --prefix="$ANDROID_NDK/../local/" --libdir="$ANDROID_NDK/../local/lib/x86/" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ./configure --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="i686-linux-android" --prefix="$ANDROID_NDK/../local/" --libdir="$ANDROID_NDK/../local/lib/x86/" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc" --enable-float
        make -j4
        make install-strip
        ;;
    linux-x86)
        ./configure --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx --prefix=/usr/local/ --libdir=/usr/local/lib32/ CC="gcc -m32"
        make -j4
        sudo make install-strip
        ./configure --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx --prefix=/usr/local/ --libdir=/usr/local/lib32/ CC="gcc -m32" --enable-float
        make -j4
        sudo make install-strip
        ;;
    linux-x86_64)
        ./configure --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx --prefix=/usr/local/ --libdir=/usr/local/lib64/
        make -j4
        sudo make install-strip
        ./configure --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx --prefix=/usr/local/ --libdir=/usr/local/lib64/ --enable-float
        make -j4
        sudo make install-strip
        ;;
    macosx-x86_64)
        ./configure --enable-shared --enable-threads --with-combined-threads --enable-sse2
        make -j4
        sudo make install-strip
        ./configure --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-float
        make -j4
        sudo make install-strip
        ;;
    windows-x86)
        unzip -o fftw-$FFTW_VERSION-dll32.zip -d fftw-$FFTW_VERSION-dll32
        # http://www.fftw.org/install/windows.html
        LIBS=(libfftw3-3 libfftw3f-3 libfftw3l-3)
        for LIB in ${LIBS[@]}; do
            lib /def:fftw-$FFTW_VERSION-dll32/$LIB.def /out:fftw-$FFTW_VERSION-dll32/$LIB.lib /machine:x86
        done
        cp -a fftw-$FFTW_VERSION-dll32/*.h $INSTALL_DIR/include
        cp -a fftw-$FFTW_VERSION-dll32/*.lib $INSTALL_DIR/lib32
        cp -a fftw-$FFTW_VERSION-dll32/*.dll $INSTALL_DIR/bin32
        ;;
    windows-x86_64)
        unzip -o fftw-$FFTW_VERSION-dll64.zip -d fftw-$FFTW_VERSION-dll64
        # http://www.fftw.org/install/windows.html
        LIBS=(libfftw3-3 libfftw3f-3 libfftw3l-3)
        for LIB in ${LIBS[@]}; do
            lib /def:fftw-$FFTW_VERSION-dll64/$LIB.def /out:fftw-$FFTW_VERSION-dll64/$LIB.lib /machine:x64
        done
        cp -a fftw-$FFTW_VERSION-dll64/*.h $INSTALL_DIR/include
        cp -a fftw-$FFTW_VERSION-dll64/*.lib $INSTALL_DIR/lib64
        cp -a fftw-$FFTW_VERSION-dll64/*.dll $INSTALL_DIR/bin64
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

if [[ $PLATFORM != windows* ]]; then
    cd ..
fi
