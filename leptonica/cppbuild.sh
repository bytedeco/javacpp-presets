#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" leptonica
    popd
    exit
fi

ZLIB=zlib-1.2.8
GIFLIB=giflib-5.1.1
LIBJPEG=libjpeg-turbo-1.4.1
LIBPNG=libpng-1.6.20
LIBTIFF=tiff-4.0.4
LIBWEBP=libwebp-0.4.3
LEPTONICA_VERSION=1.73
download http://zlib.net/$ZLIB.tar.gz $ZLIB.tar.gz
download http://downloads.sourceforge.net/project/giflib/$GIFLIB.tar.gz $GIFLIB.tar.gz
download http://downloads.sourceforge.net/project/libjpeg-turbo/1.4.1/$LIBJPEG.tar.gz $LIBJPEG.tar.gz
download http://downloads.sourceforge.net/project/libpng/libpng16/1.6.20/$LIBPNG.tar.gz $LIBPNG.tar.gz
download http://download.osgeo.org/libtiff/$LIBTIFF.tar.gz $LIBTIFF.tar.gz
download http://downloads.webmproject.org/releases/webp/$LIBWEBP.tar.gz $LIBWEBP.tar.gz
download http://www.leptonica.org/source/leptonica-$LEPTONICA_VERSION.tar.gz leptonica-$LEPTONICA_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
tar -xzvf ../$ZLIB.tar.gz
tar -xzvf ../$GIFLIB.tar.gz
tar -xzvf ../$LIBJPEG.tar.gz
tar -xzvf ../$LIBPNG.tar.gz
tar -xzvf ../$LIBTIFF.tar.gz
tar -xzvf ../$LIBWEBP.tar.gz
tar -xzvf ../leptonica-$LEPTONICA_VERSION.tar.gz

case $PLATFORM in
    android-arm)
        FLAGS="-DS_IREAD=S_IRUSR -DS_IWRITE=S_IWUSR -D__native_client__ -pthread -I$INSTALL_PATH/include/ -I$ANDROID_NDK/sources/android/cpufeatures/ --sysroot=$ANDROID_ROOT -DANDROID"
        export AR="$ANDROID_BIN-ar"
        export RANLIB="$ANDROID_BIN-ranlib"
        export CPP="$ANDROID_BIN-cpp $FLAGS"
        export CC="$ANDROID_BIN-gcc $FLAGS -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        export CXX=
        export CPPFLAGS=
        export CFLAGS=
        export CXXFLAGS=
        export LDFLAGS="-L$INSTALL_PATH/lib/ -nostdlib -Wl,--fix-cortex-a8 -z text"
        export LIBS="-lgcc -ldl -lz -lm -lc"
        export STRIP="$ANDROID_BIN-strip"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=arm-linux
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux --with-sysroot="$ANDROID_ROOT" --disable-lzma --without-x
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-androideabi --with-sysroot="$ANDROID_ROOT"
        cd src
        make -j $MAKEJ
        make install
        cd ../../leptonica-$LEPTONICA_VERSION
        patch -Np1 < ../../../leptonica-$LEPTONICA_VERSION-android.patch
        ./configure --prefix=$INSTALL_PATH --host=arm-linux-androideabi --disable-programs
        make -j $MAKEJ
        make install-strip
        ;;
     android-x86)
        FLAGS="-DS_IREAD=S_IRUSR -DS_IWRITE=S_IWUSR -pthread -I$INSTALL_PATH/include/ -I$ANDROID_NDK/sources/android/cpufeatures/ --sysroot=$ANDROID_ROOT -DANDROID"
        export AR="$ANDROID_BIN-ar"
        export RANLIB="$ANDROID_BIN-ranlib"
        export CPP="$ANDROID_BIN-cpp $FLAGS"
        export CC="$ANDROID_BIN-gcc $FLAGS -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        export CXX=
        export CPPFLAGS=
        export CFLAGS=
        export CXXFLAGS=
        export LDFLAGS="-L$INSTALL_PATH/lib/ -nostdlib -z text"
        export LIBS="-lgcc -ldl -lz -lm -lc"
        export STRIP="$ANDROID_BIN-strip"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=arm-linux
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --with-sysroot="$ANDROID_ROOT" --disable-lzma --without-x
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux-android --with-sysroot="$ANDROID_ROOT"
        cd src
        make -j $MAKEJ
        make install
        cd ../../leptonica-$LEPTONICA_VERSION
        patch -Np1 < ../../../leptonica-$LEPTONICA_VERSION-android.patch
        ./configure --prefix=$INSTALL_PATH --host=i686-linux-android --disable-programs
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86)
        export CC="$OLDCC -m32 -fPIC"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --disable-lzma
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" --disable-programs
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        export CC="$OLDCC -m64 -fPIC"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --disable-lzma
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" --disable-programs
        make -j $MAKEJ
        make install-strip
        ;;
    macosx-*)
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        [[ $ARCH == x86_64 ]] && BUILD=--build=x86_64-darwin || BUILD=
        NASM=yasm ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic $BUILD
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --disable-lzma
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        patch -Np1 < ../../../leptonica-$LEPTONICA_VERSION-macosx.patch
        ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" --disable-programs
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        export CC="gcc -m32"
        cd $ZLIB
        make -j $MAKEJ install -fwin32/Makefile.gcc BINARY_PATH=$INSTALL_PATH/bin/ INCLUDE_PATH=$INSTALL_PATH/include/ LIBRARY_PATH=$INSTALL_PATH/lib/
        cd ../$GIFLIB
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 --disable-lzma
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/ -Wl,$INSTALL_PATH/lib/*.a" --disable-programs
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86_64)
        export CC="gcc -m64"
        cd $ZLIB
        make -j $MAKEJ install -fwin32/Makefile.gcc BINARY_PATH=$INSTALL_PATH/bin/ INCLUDE_PATH=$INSTALL_PATH/include/ LIBRARY_PATH=$INSTALL_PATH/lib/
        cd ../$GIFLIB
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 --disable-lzma
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/ -Wl,$INSTALL_PATH/lib/*.a" --disable-programs
        make -j $MAKEJ
        make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
