#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" leptonica
    popd
    exit
fi

NASM_VERSION=2.14
ZLIB=zlib-1.2.11
GIFLIB=giflib-5.1.4
LIBJPEG=libjpeg-turbo-1.5.3
LIBPNG=libpng-1.6.37 # warning: libpng16 doesn't work on CentOS 6 for some reason
LIBTIFF=tiff-4.3.0
LIBWEBP=libwebp-1.2.1
LEPTONICA_VERSION=1.82.0
download https://download.videolan.org/contrib/nasm/nasm-$NASM_VERSION.tar.gz nasm-$NASM_VERSION.tar.gz
download http://zlib.net/$ZLIB.tar.gz $ZLIB.tar.gz
download http://downloads.sourceforge.net/project/giflib/$GIFLIB.tar.gz $GIFLIB.tar.gz
download http://downloads.sourceforge.net/project/libjpeg-turbo/1.5.3/$LIBJPEG.tar.gz $LIBJPEG.tar.gz
download https://sourceforge.net/projects/libpng/files/libpng16/1.6.37/$LIBPNG.tar.gz $LIBPNG.tar.gz
download http://download.osgeo.org/libtiff/$LIBTIFF.tar.gz $LIBTIFF.tar.gz
download http://downloads.webmproject.org/releases/webp/$LIBWEBP.tar.gz $LIBWEBP.tar.gz
download https://github.com/DanBloomberg/leptonica/releases/download/$LEPTONICA_VERSION/leptonica-$LEPTONICA_VERSION.tar.gz leptonica-$LEPTONICA_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../nasm-$NASM_VERSION.tar.gz
tar --totals -xzf ../$ZLIB.tar.gz
tar --totals -xzf ../$GIFLIB.tar.gz
tar --totals -xzf ../$LIBJPEG.tar.gz
tar --totals -xzf ../$LIBPNG.tar.gz
tar --totals -xzf ../$LIBTIFF.tar.gz
tar --totals -xzf ../$LIBWEBP.tar.gz
tar --totals -xzf ../leptonica-$LEPTONICA_VERSION.tar.gz

cd nasm-$NASM_VERSION
# fix for build with GCC 8.x
sedinplace 's/void pure_func/void/g' include/nasmlib.h
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ V=0
make install
export PATH=$INSTALL_PATH/bin:$PATH
cd ..

case $PLATFORM in
    android-arm)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC -DS_IREAD=S_IRUSR -DS_IWRITE=S_IWUSR -pthread -I$INSTALL_PATH/include/ -L$INSTALL_PATH/lib/ $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LDFLAGS="-ldl -lm -lc"
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
        rm contrib/arm-neon/android-ndk.c || true
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux --with-sysroot="$ANDROID_ROOT" --disable-arm-neon
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        patch -Np1 < ../../../libwebp-arm.patch
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-androideabi --with-sysroot="$ANDROID_ROOT" --disable-neon --enable-libwebpmux
        cd src
        make -j $MAKEJ
        make install
        cd ../../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux --with-sysroot="$ANDROID_ROOT" --disable-lzma --disable-zstd --without-x
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        patch -Np1 < ../../../leptonica-android.patch
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ ./configure --prefix=$INSTALL_PATH --host=arm-linux-androideabi --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
    android-arm64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC -DS_IREAD=S_IRUSR -DS_IWRITE=S_IWUSR -pthread -I$INSTALL_PATH/include/ -L$INSTALL_PATH/lib/ $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LDFLAGS="-ldl -lm -lc"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=aarch64-linux
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        rm contrib/arm-neon/android-ndk.c || true
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux --with-sysroot="$ANDROID_ROOT" --disable-arm-neon
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        patch -Np1 < ../../../libwebp-arm.patch
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux-android --with-sysroot="$ANDROID_ROOT" --disable-neon --enable-libwebpmux
        cd src
        make -j $MAKEJ
        make install
        cd ../../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux --with-sysroot="$ANDROID_ROOT" --disable-lzma --disable-zstd --without-x
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        patch -Np1 < ../../../leptonica-android.patch
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ ./configure --prefix=$INSTALL_PATH --host=aarch64-linux-android --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
     android-x86)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC -DS_IREAD=S_IRUSR -DS_IWRITE=S_IWUSR -pthread -I$INSTALL_PATH/include/ -L$INSTALL_PATH/lib/ $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LDFLAGS="-ldl -lm -lc"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=i686-linux
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
        rm contrib/arm-neon/android-ndk.c || true
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux-android --with-sysroot="$ANDROID_ROOT" --enable-libwebpmux
        cd src
        make -j $MAKEJ
        make install
        cd ../../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --with-sysroot="$ANDROID_ROOT" --disable-lzma --disable-zstd --without-x
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        patch -Np1 < ../../../leptonica-android.patch
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ ./configure --prefix=$INSTALL_PATH --host=i686-linux-android --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
     android-x86_64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC -DS_IREAD=S_IRUSR -DS_IWRITE=S_IWUSR -pthread -I$INSTALL_PATH/include/ -L$INSTALL_PATH/lib/ $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LDFLAGS="-ldl -lm -lc"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=x86_64-linux
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        rm contrib/arm-neon/android-ndk.c || true
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux-android --with-sysroot="$ANDROID_ROOT" --enable-libwebpmux
        cd src
        make -j $MAKEJ
        make install
        cd ../../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --with-sysroot="$ANDROID_ROOT" --disable-lzma --disable-zstd --without-x
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        patch -Np1 < ../../../leptonica-android.patch
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ ./configure --prefix=$INSTALL_PATH --host=x86_64-linux-android --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="gcc -m32 -fPIC"
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
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --enable-libwebpmux
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --disable-lzma --disable-zstd
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="gcc -m64 -fPIC"
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
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --enable-libwebpmux
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --disable-lzma --disable-zstd
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
    linux-armhf)
        export CFLAGS="-pthread -march=armv6 -marm -mfpu=vfp -mfloat-abi=hard -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="arm-linux-gnueabihf-gcc -fPIC"
        cd $ZLIB
        CC="arm-linux-gnueabihf-gcc -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        CC="arm-linux-gnueabihf-gcc -fPIC" ./configure --prefix=$INSTALL_PATH --host=arm-linux-gnueabihf --disable-shared
        #./configure --prefix=$INSTALL_PATH --disable-shared --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        rm contrib/arm-neon/android-ndk.c || true
        CC="arm-linux-gnueabihf-gcc -fPIC" ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" --disable-shared --with-pic --host=arm-linux-gnueabihf --disable-arm-neon
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        patch -Np1 < ../../../libwebp-arm.patch
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf --disable-neon --enable-libwebpmux
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --disable-lzma --disable-zstd --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ CC="arm-linux-gnueabihf-gcc -fPIC" ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/"  --host=arm-linux-gnueabihf --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
    linux-arm64)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="aarch64-linux-gnu-gcc -fPIC"
        cd $ZLIB
        CC="aarch64-linux-gnu-gcc -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        CC="aarch64-linux-gnu-gcc -fPIC" ./configure --prefix=$INSTALL_PATH --host=aarch64-linux-gnu --disable-shared
        #./configure --prefix=$INSTALL_PATH --disable-shared --host=aarch64-linux-gnu
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux-gnu
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        CC="aarch64-linux-gnu-gcc -fPIC" ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" --disable-shared --with-pic --host=aarch64-linux-gnu
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux-gnu --enable-libwebpmux
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --disable-lzma --disable-zstd --host=aarch64-linux-gnu
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ CC="aarch64-linux-gnu-gcc -fPIC" ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/"  --host=aarch64-linux-gnu --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
    linux-ppc64le)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          export CC="gcc -m64 -fPIC"
          export BFLAGS="--build=ppc64le-linux"
        else
          export CC="powerpc64le-linux-gnu-gcc"
          export CXX="powerpc64le-linux-gnu-g++"
          export BFLAGS="--host=powerpc64le-linux-gnu"
        fi

        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic $BFLAGS
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic $BFLAGS
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic $BFLAGS
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic $BFLAGS --enable-libwebpmux
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic $BFLAGS --disable-lzma --disable-zstd
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        sed -i s/elf64ppc/elf64lppc/ configure
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $BFLAGS --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
    macosx-*)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --enable-libwebpmux
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --disable-lzma --disable-zstd
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        patch -Np1 < ../../../leptonica-macosx.patch
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
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
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 --enable-libwebpmux
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 --disable-lzma --disable-zstd
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/ -Wl,$INSTALL_PATH/lib/libwebpmux.a -Wl,$INSTALL_PATH/lib/*.a" --build=i686-w64-mingw32 --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86_64)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
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
        cd ../$LIBWEBP
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 --enable-libwebpmux
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 --disable-lzma --disable-zstd
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/ ./configure --prefix=$INSTALL_PATH CFLAGS="-pthread -I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/ -Wl,$INSTALL_PATH/lib/libwebpmux.a -Wl,$INSTALL_PATH/lib/*.a" --build=x86_64-w64-mingw32 --disable-programs --without-libopenjpeg
        make -j $MAKEJ
        make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

# remove broken dependency_libs from files for libtool
sedinplace '/dependency_libs/d' ../lib/*.la || true

cd ../..
