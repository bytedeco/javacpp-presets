#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tesseract
    popd
    exit
fi

TESSERACT_VERSION=5.0.1
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
sedinplace '/tiff/d' Makefile.am
sedinplace '/strcmp(locale, "C")/d' src/api/baseapi.cpp
bash autogen.sh
chmod 755 configure config/install-sh
export AUTOCONF=:
export AUTOHEADER=:
export AUTOMAKE=:
export ACLOCAL=:

# Disable external dependencies on asciidoc, libarchive and libtiff
sedinplace 's/ac_cv_prog_have_asciidoc="true"/ac_cv_prog_have_asciidoc="false"/g' configure
sedinplace 's/"libarchive"//g' configure
sedinplace 's/-ltiff//g' Makefile.in

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
        patch -Np1 < ../../../tesseract-android.patch
        sedinplace 's/avx=true/avx=false/g' configure
        sedinplace 's/avx2=true/avx2=false/g' configure
        sedinplace 's/fma=true/fma=false/g' configure
        sedinplace 's/sse41=true/sse41=false/g' configure
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" AR="$ANDROID_PREFIX-ar" RANLIB="$ANDROID_PREFIX-ranlib" CC="$ANDROID_CC $ANDROID_FLAGS" CXX="$ANDROID_CC++ $ANDROID_FLAGS" STRIP="$ANDROID_PREFIX-strip" CPPFLAGS="-I$LEPTONICA_PATH/include/ -Wno-c++11-narrowing" LDFLAGS="-L$LEPTONICA_PATH/lib/" LIBS="-llept $ANDROID_LIBS"
        # Disable what Autoconf tries to do but that fails for Android
        sedinplace '/predep_objects=/d' libtool
        sedinplace '/postdep_objects=/d' libtool
        sedinplace '/predeps=/d' libtool
        sedinplace '/postdeps=/d' libtool
        sedinplace 's/-nostdlib //g' libtool
        chmod -w libtool
        make -j $MAKEJ
        make install-strip
        ;;
    android-arm64)
        patch -Np1 < ../../../tesseract-android.patch
        sedinplace 's/avx=true/avx=false/g' configure
        sedinplace 's/avx2=true/avx2=false/g' configure
        sedinplace 's/fma=true/fma=false/g' configure
        sedinplace 's/sse41=true/sse41=false/g' configure
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --host="aarch64-linux-android" --with-sysroot="$ANDROID_ROOT" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" AR="$ANDROID_PREFIX-ar" RANLIB="$ANDROID_PREFIX-ranlib" CC="$ANDROID_CC $ANDROID_FLAGS" CXX="$ANDROID_CC++ $ANDROID_FLAGS" STRIP="$ANDROID_PREFIX-strip" CPPFLAGS="-I$LEPTONICA_PATH/include/ -Wno-c++11-narrowing" LDFLAGS="-L$LEPTONICA_PATH/lib/" LIBS="-llept $ANDROID_LIBS"
        # Disable what Autoconf tries to do but that fails for Android
        sedinplace '/predep_objects=/d' libtool
        sedinplace '/postdep_objects=/d' libtool
        sedinplace '/predeps=/d' libtool
        sedinplace '/postdeps=/d' libtool
        sedinplace 's/-nostdlib //g' libtool
        chmod -w libtool
        make -j $MAKEJ
        make install-strip
        ;;
    android-x86)
        patch -Np1 < ../../../tesseract-android.patch
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" AR="$ANDROID_PREFIX-ar" RANLIB="$ANDROID_PREFIX-ranlib" CC="$ANDROID_CC $ANDROID_FLAGS" CXX="$ANDROID_CC++ $ANDROID_FLAGS" STRIP="$ANDROID_PREFIX-strip" CPPFLAGS="-I$LEPTONICA_PATH/include/ -Wno-c++11-narrowing" LDFLAGS="-L$LEPTONICA_PATH/lib/" LIBS="-llept $ANDROID_LIBS"
        # Disable what Autoconf tries to do but that fails for Android
        sedinplace '/predep_objects=/d' libtool
        sedinplace '/postdep_objects=/d' libtool
        sedinplace '/predeps=/d' libtool
        sedinplace '/postdeps=/d' libtool
        sedinplace 's/-nostdlib //g' libtool
        chmod -w libtool
        make -j $MAKEJ
        make install-strip
        ;;
    android-x86_64)
        patch -Np1 < ../../../tesseract-android.patch
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --host="x86_64-linux-android" --with-sysroot="$ANDROID_ROOT" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" AR="$ANDROID_PREFIX-ar" RANLIB="$ANDROID_PREFIX-ranlib" CC="$ANDROID_CC $ANDROID_FLAGS" CXX="$ANDROID_CC++ $ANDROID_FLAGS" STRIP="$ANDROID_PREFIX-strip" CPPFLAGS="-I$LEPTONICA_PATH/include/ -Wno-c++11-narrowing" LDFLAGS="-L$LEPTONICA_PATH/lib/" LIBS="-llept $ANDROID_LIBS"
        # Disable what Autoconf tries to do but that fails for Android
        sedinplace '/predep_objects=/d' libtool
        sedinplace '/postdep_objects=/d' libtool
        sedinplace '/predeps=/d' libtool
        sedinplace '/postdeps=/d' libtool
        sedinplace 's/-nostdlib //g' libtool
        chmod -w libtool
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86)
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH CC="gcc -m32" CXX="g++ -m32" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-armhf)
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --host=arm-linux-gnueabihf CC="arm-linux-gnueabihf-gcc" CXX="arm-linux-gnueabihf-g++" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-arm64)
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --host=aarch64-linux-gnu CC="aarch64-linux-gnu-gcc" CXX="aarch64-linux-gnu-g++" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH CC="gcc -m64" CXX="g++ -m64" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-ppc64le)
        patch -Np1 < ../../../tesseract-openmp.patch
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          PKG_CONFIG= ./configure --prefix=$INSTALL_PATH CC="gcc -m64" CXX="g++ -m64" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        else
          PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --host=powerpc64le-linux-gnu CC=powerpc64le-linux-gnu-gcc CXX=powerpc64le-linux-gnu-g++ LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        fi
        make -j $MAKEJ
        make install-strip
        ;;
    macosx-*)
        sedinplace 's/\\$rpath/@rpath/g' configure
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/ -Wl,-rpath,$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        # cp src/vs2010/port/* src/ccutil/
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --host="i686-w64-mingw32" CC="gcc -m32" CXX="g++ -m32 -fpermissive" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86_64)
        # cp src/vs2010/port/* src/ccutil/
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --host="x86_64-w64-mingw32" CC="gcc -m64" CXX="g++ -m64 -fpermissive" LEPTONICA_CFLAGS="-I$LEPTONICA_PATH/include/leptonica/" LEPTONICA_LIBS="-L$LEPTONICA_PATH/lib/ -llept" CPPFLAGS="-I$LEPTONICA_PATH/include/" LDFLAGS="-L$LEPTONICA_PATH/lib/" LIBS="-llept"
        make -j $MAKEJ
        make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
