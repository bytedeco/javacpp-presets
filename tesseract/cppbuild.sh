#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tesseract
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    TESSERACT_VERSION=3.03-0.1.rc1
    [[ $PLATFORM == *64 ]] && BITS=64 || BITS=32
    download http://mirrors.kernel.org/fedora/releases/21/Everything/x86_64/os/Packages/m/mingw$BITS-tesseract-$TESSERACT_VERSION.fc21.noarch.rpm mingw$BITS-tesseract-$TESSERACT_VERSION.fc21.noarch.rpm

    function extract {
        /C/Program\ Files/7-Zip/7z x -y $1
    }
    extract mingw$BITS-tesseract-$TESSERACT_VERSION.fc21.noarch.rpm

    mkdir -p $PLATFORM
    cd $PLATFORM
    rm -Rf include lib bin
    extract ../mingw$BITS-tesseract-$TESSERACT_VERSION.fc21.noarch.cpio
else
    TESSERACT_VERSION=3.03
    download "https://drive.google.com/uc?export=download&id=0B7l10Bj_LprhSGN2bTYwemVRREU" tesseract-$TESSERACT_VERSION.tar.gz

    mkdir -p $PLATFORM
    cd $PLATFORM
    INSTALL_PATH=`pwd`
    tar -xzvf ../tesseract-$TESSERACT_VERSION.tar.gz
    cd tesseract-$TESSERACT_VERSION
fi

case $PLATFORM in
    android-arm)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-android.patch
        cp "$ANDROID_ROOT/usr/lib/crtbegin_so.o" "$ANDROID_ROOT/usr/lib/crtend_so.o" api
        ar r api/librt.a "$ANDROID_ROOT/usr/lib/crtbegin_dynamic.o"
        ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" LIBLEPT_HEADERSDIR="$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/include/" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/include/ -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/armeabi/include/ -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/armeabi/ -nostdlib -Wl,--fix-cortex-a8 -L$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/lib/ -L./" LIBS="-lgnustl_static -lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ;;
     android-x86)
        patch -Np1 < ../../../tesseract-$TESSERACT_VERSION-android.patch
        cp "$ANDROID_ROOT/usr/lib/crtbegin_so.o" "$ANDROID_ROOT/usr/lib/crtend_so.o" api
        ar r api/librt.a "$ANDROID_ROOT/usr/lib/crtbegin_dynamic.o"
        ./configure --prefix=$INSTALL_PATH --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" LIBLEPT_HEADERSDIR="$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/include/" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/include/ -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/x86/include/ -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300 -L$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/lib/" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/x86/ -nostdlib -L$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/lib/ -L." LIBS="-lgnustl_static -lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ;;
    linux-x86)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m32" CXX="g++ -m32" LIBLEPT_HEADERSDIR="$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/include/" CPPFLAGS="-I$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/include/" LDFLAGS="-L$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/lib/"
        make -j4
        make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m64" CXX="g++ -m64" LIBLEPT_HEADERSDIR="$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/include/" CPPFLAGS="-I$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/include/" LDFLAGS="-L$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/lib/"
        make -j4
        make install-strip
        ;;
    macosx-*)
        ./configure --prefix=$INSTALL_PATH LIBLEPT_HEADERSDIR="$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/include/" CPPFLAGS="-I$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/include/" LDFLAGS="-L$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/lib/"
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
