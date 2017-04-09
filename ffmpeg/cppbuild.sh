#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" ffmpeg
    popd
    exit
fi

DISABLE="--disable-w32threads --disable-iconv --disable-libxcb --disable-opencl --disable-sdl --disable-bzlib --disable-lzma"
ENABLE="--enable-pthreads --enable-shared --enable-gpl --enable-version3 --enable-nonfree --enable-runtime-cpudetect --enable-zlib --enable-libmp3lame --enable-libspeex --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-openssl --enable-libopenh264 --enable-libx264 --enable-libx265 --enable-libvpx"

# minimal configuration to support MPEG-4 streams with H.264 and AAC as well as Motion JPEG
# DISABLE="--disable-w32threads --disable-iconv --disable-libxcb --disable-opencl --disable-sdl --disable-bzlib --disable-lzma --disable-everything"
# ENABLE="--enable-pthreads --enable-shared --enable-runtime-cpudetect --enable-libopenh264 --enable-encoder=libopenh264 --enable-encoder=aac --enable-encoder=mjpeg --enable-decoder=h264 --enable-decoder=aac --enable-decoder=mjpeg --enable-parser=h264 --enable-parser=aac --enable-parser=mjpeg --enable-muxer=mp4 --enable-muxer=rtsp --enable-muxer=mjpeg --enable-demuxer=mov --enable-demuxer=rtsp --enable-demuxer=mjpeg --enable-protocol=file --enable-protocol=http --enable-protocol=rtp --enable-protocol=rtmp"

ZLIB=zlib-1.2.11
LAME=lame-3.99.5
SPEEX=speex-1.2rc2
OPENCORE_AMR=opencore-amr-0.1.3
OPENSSL=openssl-1.0.2k
OPENH264_VERSION=1.6.0
X265=x265_2.3
VPX_VERSION=v1.6.1
ALSA_VERSION=1.1.3
FFMPEG_VERSION=3.2.4
download http://zlib.net/$ZLIB.tar.gz $ZLIB.tar.gz
download http://downloads.sourceforge.net/project/lame/lame/3.99/$LAME.tar.gz $LAME.tar.gz
download http://downloads.xiph.org/releases/speex/$SPEEX.tar.gz $SPEEX.tar.gz
download http://sourceforge.net/projects/opencore-amr/files/opencore-amr/$OPENCORE_AMR.tar.gz/download $OPENCORE_AMR.tar.gz
download https://www.openssl.org/source/$OPENSSL.tar.gz $OPENSSL.tar.gz
download https://github.com/cisco/openh264/archive/v$OPENH264_VERSION.tar.gz openh264-$OPENH264_VERSION.tar.gz
download ftp://ftp.videolan.org/pub/videolan/x264/snapshots/last_stable_x264.tar.bz2 last_stable_x264.tar.bz2
download https://ftp.videolan.org/pub/videolan/x265/$X265.tar.gz $X265.tar.gz
download https://chromium.googlesource.com/webm/libvpx/+archive/$VPX_VERSION.tar.gz libvpx-$VPX_VERSION.tar.gz
download ftp://ftp.alsa-project.org/pub/lib/alsa-lib-$ALSA_VERSION.tar.bz2 alsa-lib-$ALSA_VERSION.tar.bz2
download http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 ffmpeg-$FFMPEG_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../$ZLIB.tar.gz
tar --totals -xzf ../$LAME.tar.gz
tar --totals -xzf ../$SPEEX.tar.gz
tar --totals -xzf ../$OPENCORE_AMR.tar.gz
tar --totals -xzf ../$OPENSSL.tar.gz
tar --totals -xzf ../openh264-$OPENH264_VERSION.tar.gz
tar --totals -xjf ../last_stable_x264.tar.bz2
tar --totals -xzf ../$X265.tar.gz
mkdir -p libvpx-$VPX_VERSION
tar --totals -xzf ../libvpx-$VPX_VERSION.tar.gz -C libvpx-$VPX_VERSION
tar --totals -xjf ../ffmpeg-$FFMPEG_VERSION.tar.bz2
X264=`echo x264-snapshot-*`

case $PLATFORM in
    android-arm)
        export AR="$ANDROID_BIN-ar"
        export CPP="$ANDROID_BIN-cpp"
        export CC="$ANDROID_BIN-gcc"
        export CXX="$ANDROID_BIN-g++"
        export RANLIB="$ANDROID_BIN-ranlib"
        export STRIP="$ANDROID_BIN-strip"
        export CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID"
        export CFLAGS="$CPPFLAGS -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        export CXXFLAGS="$CFLAGS"
        export LDFLAGS="-nostdlib -Wl,--fix-cortex-a8 -z text"
        export LIBS="-lgcc -ldl -lz -lm -lc"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=arm-linux
        make -j $MAKEJ
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux
        make -j $MAKEJ
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux
        cd libspeex
        make -j $MAKEJ
        make install
        cd ../include
        make install
        cd ../../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux
        make -j $MAKEJ
        make install
        cd ../$OPENSSL
        ./Configure --prefix=$INSTALL_PATH android-armv7 $CFLAGS no-shared
        ANDROID_DEV="$ANDROID_ROOT/usr" make # fails with -j > 1
        make install
        cd ../openh264-$OPENH264_VERSION
        LDFLAGS= make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=arm USE_ASM=No NDKROOT="$ANDROID_NDK" TARGET="$ANDROID_ROOT" libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=arm-linux --extra-cflags="$CFLAGS" --extra-ldflags="$LDFLAGS $LIBS"
        make -j $MAKEJ
        make install
        cd ../$X265
        patch -Np1 < ../../../$X265-android.patch || true
        $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DCMAKE_TOOLCHAIN_FILE=android-arm.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ x265-static
        make install
        cd ../libvpx-$VPX_VERSION
        LDFLAGS= ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --sdk-path=$ANDROID_NDK --target=armv7-android-gcc --disable-runtime-cpu-detect --disable-neon --disable-neon-asm
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-android.patch
        ./configure --prefix=.. $DISABLE $ENABLE --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --ranlib="$ANDROID_BIN-ranlib" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=arm --extra-cflags="-I../include/ $CFLAGS" --extra-ldflags="$ANDROID_ROOT/usr/lib/crtbegin_so.o -L../lib/ -L$ANDROID_CPP/libs/armeabi/ $LDFLAGS" --extra-libs="-lgnustl_static $LIBS" --disable-symver --disable-programs
        make -j $MAKEJ
        make install
        ;;

     android-x86)
        export AR="$ANDROID_BIN-ar"
        export CPP="$ANDROID_BIN-cpp"
        export CC="$ANDROID_BIN-gcc"
        export CXX="$ANDROID_BIN-g++"
        export RANLIB="$ANDROID_BIN-ranlib"
        export STRIP="$ANDROID_BIN-strip"
        export CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID"
        export CFLAGS="$CPPFLAGS -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        export CXXFLAGS="$CFLAGS"
        export LDFLAGS="-nostdlib -z text"
        export LIBS="-lgcc -ldl -lz -lm -lc"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=i686-linux
        make -j $MAKEJ
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        cd libspeex
        make -j $MAKEJ
        make install
        cd ../include
        make install
        cd ../../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../$OPENSSL
        ./Configure --prefix=$INSTALL_PATH android-x86 $CFLAGS no-shared
        ANDROID_DEV="$ANDROID_ROOT/usr" make # fails with -j > 1
        make install
        cd ../openh264-$OPENH264_VERSION
        LDFLAGS= make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=x86 USE_ASM=No NDKROOT="$ANDROID_NDK" TARGET="$ANDROID_ROOT" libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=i686-linux --disable-asm --extra-cflags="$CFLAGS" --extra-ldflags="$LDFLAGS $LIBS"
        make -j $MAKEJ
        make install
        cd ../$X265
        patch -Np1 < ../../../$X265-android.patch || true
        $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DCMAKE_TOOLCHAIN_FILE=android-x86.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ x265-static
        make install
        cd ../libvpx-$VPX_VERSION
        ASFLAGS="-D__ANDROID__" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-tools --target=x86-android-gcc
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-android.patch
        ./configure --prefix=.. $DISABLE $ENABLE --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --ranlib="$ANDROID_BIN-ranlib" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=atom --extra-cflags="-I../include/ $CFLAGS" --extra-ldflags="$ANDROID_ROOT/usr/lib/crtbegin_so.o -L../lib/ -L$ANDROID_CPP/libs/x86/ $LDFLAGS" --extra-libs="-lgnustl_static $LIBS" --disable-symver --disable-programs
        make -j $MAKEJ
        make install
        ;;

    linux-x86)
        cd $ZLIB
        CC="gcc -m32 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32 -msse2"
        make -j $MAKEJ
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32"
        make -j $MAKEJ
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32" CXXFLAGS="-m32"
        make -j $MAKEJ
        make install
        cd ../$OPENSSL
        ./Configure linux-elf -m32 -fPIC no-shared --prefix=$INSTALL_PATH
        make # fails with -j > 1
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86 USE_ASM=No libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../$X265
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --target=x86-linux-gcc --as=yasm
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        [[ $ENABLE =~ "--enable-gpl" ]] && X11GRAB="--enable-x11grab" || X11GRAB=
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-linux.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $X11GRAB --cc="gcc -m32" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl"
        make -j $MAKEJ
        make install
        ;;

    linux-x86_64)
        cd $ZLIB
        CC="gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64" CXXFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../$OPENSSL
        ./Configure linux-x86_64 -fPIC no-shared --prefix=$INSTALL_PATH
        make # fails with -j > 1
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86_64 USE_ASM=No libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=x86_64-linux
        make -j $MAKEJ
        make install
        cd ../$X265
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --target=x86_64-linux-gcc --as=yasm
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        [[ $ENABLE =~ "--enable-gpl" ]] && X11GRAB="--enable-x11grab" || X11GRAB=
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-linux.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $X11GRAB --cc="gcc -m64" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl"
        make -j $MAKEJ
        make install
        ;;

    linux-armhf)
        tar --totals -xjf ../alsa-lib-$ALSA_VERSION.tar.bz2

        export CFLAGS="-march=armv6 -marm -mfpu=vfp -mfloat-abi=hard"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        HOST_ARCH="$(uname -m)"
        CROSSCOMPILE=1
        if [[ $HOST_ARCH == *"arm"* ]]
        then
          echo "Detected arm arch so not cross compiling";
          CROSSCOMPILE=0
        else
          echo "Detected non arm arch so cross compiling";
        fi

        cd $ZLIB
        CC="arm-linux-gnueabihf-gcc -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install
        cd ../$OPENSSL
        if [ $CROSSCOMPILE -eq 1 ]
        then
          ./Configure linux-generic32 -march=armv6 -mfpu=vfp -mfloat-abi=hard -fPIC no-shared --prefix=$INSTALL_PATH --cross-compile-prefix=arm-linux-gnueabihf-
        else
          ./Configure linux-generic32 -fPIC no-shared --prefix=$INSTALL_PATH
        fi
        make # fails with -j > 1
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=arm-linux-gnueabihf-ar ARCH=armhf USE_ASM=No libraries install-static CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++
        cd ../$X264
        if [ $CROSSCOMPILE -eq 1 ]
        then
          ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --disable-cli --host=arm-linux-gnueabihf --cross-prefix="arm-linux-gnueabihf-" --disable-asm
        else
          ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --disable-cli --disable-asm
        fi
        make -j $MAKEJ
        make install
        cd ../$X265
        if [ $CROSSCOMPILE -eq 1 ]
        then
          $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_STRIP=arm-linux-gnueabihf-strip -DCMAKE_FIND_ROOT_PATH=arm-linux-gnueabihf -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        else
          $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        fi
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        if [ $CROSSCOMPILE -eq 1 ]
        then
          CROSS=arm-linux-gnueabihf- ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --target=armv7-linux-gcc
        else
          ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples
        fi
        make -j $MAKEJ
        make install
        cd ../alsa-lib-$ALSA_VERSION/
        ./configure --host=arm-linux-gnueabihf --prefix=$INSTALL_PATH --disable-python
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        [[ $ENABLE =~ "--enable-gpl" ]] && X11GRAB="--enable-x11grab" || X11GRAB=
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-linux.patch
        if [ $CROSSCOMPILE -eq 1 ]
        then
          PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --cc="arm-linux-gnueabihf-gcc" --extra-cflags="-I/opt/additionalInclude/ -I../include/" --extra-ldflags="-L../lib/ -L/opt/additionalLib" --extra-libs="-lstdc++ -ldl -lasound" --enable-cross-compile --arch=armhf --target-os=linux --cross-prefix="arm-linux-gnueabihf-" --pkg-config-flags="--static" --pkg-config="pkg-config --static" --disable-doc --disable-programs
        else
          PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl -lasound" --pkg-config-flags="--static" --pkg-config="pkg-config --static"
        fi
        make -j $MAKEJ
        make install
        ;;

    linux-ppc64le)
        cd $ZLIB
        CC="gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64" CXXFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../$OPENSSL
        ./Configure linux-ppc64le -fPIC no-shared --prefix=$INSTALL_PATH
        make # fails with -j > 1
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=ppc64le USE_ASM=No libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=ppc64le-linux
        make -j $MAKEJ
        make install
        cd ../$X265
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --target=generic-gnu
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        [[ $ENABLE =~ "--enable-gpl" ]] && X11GRAB="--enable-x11grab" || X11GRAB=
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-linux.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $X11GRAB --cc="gcc -m64" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl"
        make -j $MAKEJ
        make install
        ;;

    macosx-*)
        cd $ZLIB
        CC="clang -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../$OPENSSL
        ./Configure darwin64-x86_64-cc -fPIC no-shared --prefix=$INSTALL_PATH
        make # fails with -j > 1
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar USE_ASM=No libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl
        make -j $MAKEJ
        make install
        cd ../$X265
        CC="clang" CXX="clang++" $CMAKE -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-macosx.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-indev=avfoundation --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl" --disable-doc --disable-programs
        make -j $MAKEJ
        make install
        ;;

    windows-x86)
        cd $ZLIB
        make -j $MAKEJ install -fwin32/Makefile.gcc BINARY_PATH=$INSTALL_PATH/bin/ INCLUDE_PATH=$INSTALL_PATH/include/ LIBRARY_PATH=$INSTALL_PATH/lib/
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 CFLAGS="-m32 -msse2"
        make -j $MAKEJ
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 CFLAGS="-m32"
        make -j $MAKEJ
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 CFLAGS="-m32" CXXFLAGS="-m32"
        make -j $MAKEJ
        make install
        cd ../$OPENSSL
        ./Configure mingw -fPIC no-shared --prefix=$INSTALL_PATH
        make # fails with -j > 1
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86 USE_ASM=No libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --disable-win32thread --host=i686-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../$X265
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -G "MSYS Makefiles" -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --target=x86-win32-gcc
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-windows.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-indev=dshow --target-os=mingw32 --cc="gcc -m32" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lgcc -lgcc_eh -lpthread -Wl,-Bdynamic"
        make -j $MAKEJ
        make install
        ;;

    windows-x86_64)
        cd $ZLIB
        make -j $MAKEJ install -fwin32/Makefile.gcc BINARY_PATH=$INSTALL_PATH/bin/ INCLUDE_PATH=$INSTALL_PATH/include/ LIBRARY_PATH=$INSTALL_PATH/lib/
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 CFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 CFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 CFLAGS="-m64" CXXFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../$OPENSSL
        ./Configure mingw64 -fPIC no-shared --prefix=$INSTALL_PATH
        make # fails with -j > 1
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86_64 USE_ASM=No libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --disable-win32thread --host=x86_64-w64-mingw32
        make -j $MAKEJ
        make install
        cd ../$X265
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -G "MSYS Makefiles" -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --target=x86_64-win64-gcc
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-windows.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-indev=dshow --target-os=mingw32 --cc="gcc -m64" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lgcc -lgcc_eh -lpthread -Wl,-Bdynamic"
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..

