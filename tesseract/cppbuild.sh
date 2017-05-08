#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tesseract
    popd
    exit
fi

TESSERACT_VERSION=3.05.00
download https://github.com/tesseract-ocr/tesseract/archive/$TESSERACT_VERSION.tar.gz tesseract-$TESSERACT_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../tesseract-$TESSERACT_VERSION.tar.gz
cd tesseract-$TESSERACT_VERSION
if [[ "${ACLOCAL_PATH:-}" == C:\\msys64\\* ]]; then
    export ACLOCAL_PATH=/mingw64/share/aclocal:/usr/share/aclocal
fi
bash autogen.sh

LEPTONICA_PATH=$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -d "$P/include/leptonica" ]]; then
            LEPTONICA_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

LEPTONICA_PATH="${LEPTONICA_PATH//\\//}"

case $PLATFORM in
    android-arm)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-android.patch
        cp "$ANDROID_ROOT/usr/lib/crtbegin_so.o" "$ANDROID_ROOT/usr/lib/crtend_so.o" api
        "$ANDROID_BIN-ar" r api/librt.a "$ANDROID_ROOT/usr/lib/crtbegin_dynamic.o"
        ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" LIBLEPT_HEADERSDIR="$LEPTONICA_PATH/include/" AR="$ANDROID_BIN-ar" RANLIB="$ANDROID_BIN-ranlib" CPP="$ANDROID_BIN-cpp" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$LEPTONICA_PATH/include/ -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/armeabi/include/ -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/armeabi/ -nostdlib -Wl,--fix-cortex-a8 -z text -L$LEPTONICA_PATH/lib/ -L./" LIBS="-lgnustl_static -lgcc -ldl -lz -lm -lc"
        make -j $MAKEJ
        make install-strip
        ;;
     android-x86)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-android.patch
        cp "$ANDROID_ROOT/usr/lib/crtbegin_so.o" "$ANDROID_ROOT/usr/lib/crtend_so.o" api
        "$ANDROID_BIN-ar" r api/librt.a "$ANDROID_ROOT/usr/lib/crtbegin_dynamic.o"
        ./configure --prefix=$INSTALL_PATH --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" LIBLEPT_HEADERSDIR="$LEPTONICA_PATH/include/" AR="$ANDROID_BIN-ar" RANLIB="$ANDROID_BIN-ranlib" CPP="$ANDROID_BIN-cpp" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$LEPTONICA_PATH/include/ -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/x86/include/ -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300 -L$LEPTONICA_PATH/lib/" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/x86/ -nostdlib -z text -L$LEPTONICA_PATH/lib/ -L." LIBS="-lgnustl_static -lgcc -ldl -lz -lm -lc"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-linux.patch
        ./configure --prefix=$INSTALL_PATH CC="$OLDCC -m32" CXX="$OLDCXX -m32" LIBLEPT_HEADERSDIR="$LEPTONICA_PATH/include/" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-armhf)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-linux.patch
        ./configure --prefix=$INSTALL_PATH --host=arm-linux-gnueabihf CC="arm-linux-gnueabihf-gcc" CXX="arm-linux-gnueabihf-g++" LIBLEPT_HEADERSDIR="$LEPTONICA_PATH/include/" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-arm64)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-linux.patch
        ./configure --prefix=$INSTALL_PATH --host=aarch64-linux-gnu CC="aarch64-linux-gnu-gcc" CXX="aarch64-linux-gnu-g++" LIBLEPT_HEADERSDIR="$LEPTONICA_PATH/include/" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-linux.patch
        ./configure --prefix=$INSTALL_PATH CC="$OLDCC -m64" CXX="$OLDCXX -m64" LIBLEPT_HEADERSDIR="$LEPTONICA_PATH/include/" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-ppc64le)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-linux.patch
        ./configure --prefix=$INSTALL_PATH CC="$OLDCC -m64" CXX="$OLDCXX -m64" LIBLEPT_HEADERSDIR="$LEPTONICA_PATH/include/" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/"
        make -j $MAKEJ
        make install-strip
        ;;
    macosx-*)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-macosx.patch
        ./configure --prefix=$INSTALL_PATH LIBLEPT_HEADERSDIR="$LEPTONICA_PATH/include/" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/"
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-windows.patch
        cp vs2010/port/* ccutil/
        ./configure --prefix=$INSTALL_PATH --host="i686-w64-mingw32" CC="gcc -m32" CXX="g++ -m32 -fpermissive" LIBLEPT_HEADERSDIR="$LEPTONICA_PATH/include/" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/"
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86_64)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-windows.patch
        cp vs2010/port/* ccutil/
        ./configure --prefix=$INSTALL_PATH --host="x86_64-w64-mingw32" CC="gcc -m64" CXX="g++ -m64 -fpermissive" LIBLEPT_HEADERSDIR="$LEPTONICA_PATH/include/" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/"
        make -j $MAKEJ
        make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
