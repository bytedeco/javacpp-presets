#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tesseract
    popd
    exit
fi

TESSERACT_VERSION=4.0.0
AUTOCONF=autoconf-archive-2017.09.28

download http://gnu.uberglobalmirror.com/autoconf-archive/$AUTOCONF.tar.xz $AUTOCONF.tar.xz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

echo "Decompressing autoconf..."
tar --totals -xzf ../$AUTOCONF.tar.xz

# As this comes to us uncompressed we skip the TAR step but have to put the file in $PLATFORM
# Check if folder already exists, if so pull the latest changes
# Unfortunately there isn't an official Tesseract 4.0.0 release, so as Tesseract recommends
# we use their 'master' branch, although this may cause breaking changes in presets.
if [ ! -d tesseract-$TESSERACT_VERSION ] ; then
    git clone https://github.com/tesseract-ocr/tesseract.git tesseract-$TESSERACT_VERSION
    cd tesseract-$TESSERACT_VERSION
else
    cd tesseract-$TESSERACT_VERSION
    # pull from remote
    git pull
fi


if [[ "${ACLOCAL_PATH:-}" == C:\\msys64\\* ]]; then
    export ACLOCAL_PATH=/mingw64/share/aclocal:/usr/share/aclocal
fi

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

# go to the AUTOCONF dir, as we all need to build autoconf
cd ../$AUTOCONF

case $PLATFORM in
    android-arm)
        ./configure --prefix=$INSTALL_PATH --uname=arm-linux
        make -j $MAKEJ
        make install
        cd ../tesseract-$TESSERACT_VERSION
        bash autogen.sh
        patch -Np1 < ../../../tesseract-android.patch
        cp "$ANDROID_ROOT/usr/lib/crtbegin_so.o" "$ANDROID_ROOT/usr/lib/crtend_so.o" api
        "$ANDROID_BIN-ar" r api/librt.a "$ANDROID_ROOT/usr/lib/crtbegin_dynamic.o"
        ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" AR="$ANDROID_BIN-ar" RANLIB="$ANDROID_BIN-ranlib" CPP="$ANDROID_BIN-cpp" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$LEPTONICA_PATH/include/ -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/armeabi/include/ -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/armeabi/ -nostdlib -Wl,--fix-cortex-a8 -z text -L$LEPTONICA_PATH/lib/ -L./" LIBS="-llept -lgnustl_static -lgcc -ldl -lz -lm -lc"
        make -j $MAKEJ
        make install-strip
        ;;
     android-x86)
        ./configure --prefix=$INSTALL_PATH --uname=i686-linux
        make -j $MAKEJ
        make install
        cd ../tesseract-$TESSERACT_VERSION
        bash autogen.sh
        patch -Np1 < ../../../tesseract-android.patch
        cp "$ANDROID_ROOT/usr/lib/crtbegin_so.o" "$ANDROID_ROOT/usr/lib/crtend_so.o" api
        "$ANDROID_BIN-ar" r api/librt.a "$ANDROID_ROOT/usr/lib/crtbegin_dynamic.o"
        ./configure --prefix=$INSTALL_PATH --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" AR="$ANDROID_BIN-ar" RANLIB="$ANDROID_BIN-ranlib" CPP="$ANDROID_BIN-cpp" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$LEPTONICA_PATH/include/ -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/x86/include/ -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300 -L$LEPTONICA_PATH/lib/" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/x86/ -nostdlib -z text -L$LEPTONICA_PATH/lib/ -L." LIBS="-llept -lgnustl_static -lgcc -ldl -lz -lm -lc"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86)
        ./configure --prefix=$INSTALL_PATH
        make -j $MAKEJ
        make install
        cd ../tesseract-$TESSERACT_VERSION
        bash autogen.sh
        #patch -Np1 < ../../../tesseract-linux.patch
        ./configure --prefix=$INSTALL_PATH CC="$OLDCC -m32" CXX="$OLDCXX -m32" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-armhf)
        CC="arm-linux-gnueabihf-gcc -fPIC" ./configure --prefix=$INSTALL_PATH
        make -j $MAKE
        make install
        cd ../tesseract-$TESSERACT_VERSION
        bash autogen.sh
        #patch -Np1 < ../../../tesseract-linux.patch
        ./configure --prefix=$INSTALL_PATH --host=arm-linux-gnueabihf CC="arm-linux-gnueabihf-gcc" CXX="arm-linux-gnueabihf-g++" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-arm64)
        CC="aarch64-linux-gnu-gcc -fPIC" ./configure --prefix=$INSTALL_PATH --host=aarch64-linux-gnu --disable-shared
        make -j $MAKE
        make install
        cd ../tesseract-$TESSERACT_VERSION
        bash autogen.sh
        #patch -Np1 < ../../../tesseract-linux.patch
        ./configure --prefix=$INSTALL_PATH --host=aarch64-linux-gnu CC="aarch64-linux-gnu-gcc" CXX="aarch64-linux-gnu-g++" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH
        make -j $MAKEJ
        make install
        cd ../tesseract-$TESSERACT_VERSION
        bash autogen.sh
        #patch -Np1 < ../../../tesseract-linux.patch
        ./configure --prefix=$INSTALL_PATH CC="$OLDCC -m64" CXX="$OLDCXX -m64" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-ppc64le)
        ./configure --prefix=$INSTALL_PATH
        make -j $MAKEJ
        make install
        cd ../tesseract-$TESSERACT_VERSION
        bash autogen.sh
        #patch -Np1 < ../../../tesseract-linux.patch
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH CC="$OLDCC -m64" CXX="$OLDCXX -m64" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        else
          ./configure --prefix=$INSTALL_PATH --host=powerpc64le-linux-gnu CC=powerpc64le-linux-gnu-gcc CXX=powerpc64le-linux-gnu-g++ LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        fi
        make -j $MAKEJ
        make install-strip
        ;;
    macosx-*)
        ./configure --prefix=$INSTALL_PATH
        make -j $MAKEJ
        make install
        cd ../tesseract-$TESSERACT_VERSION
        bash autogen.sh
        patch -Np1 < ../../../tesseract-macosx.patch
        ./configure --prefix=$INSTALL_PATH LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/ -std=c++11" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../tesseract-$TESSERACT_VERSION
        bash autogen.sh
        patch -Np1 < ../../../tesseract-windows.patch
        cp vs2010/port/* ccutil/
        ./configure --prefix=$INSTALL_PATH --host="i686-w64-mingw32" CC="gcc -m32" CXX="g++ -m32 -fpermissive" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86_64)
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../tesseract-$TESSERACT_VERSION
        bash autogen.sh
        patch -Np1 < ../../../tesseract-windows.patch
        cp vs2010/port/* ccutil/
        ./configure --prefix=$INSTALL_PATH --host="x86_64-w64-mingw32" CC="gcc -m64" CXX="g++ -m64 -fpermissive" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..