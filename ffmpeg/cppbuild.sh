#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" ffmpeg
    popd
    exit
fi

DISABLE="--disable-iconv --disable-opencl --disable-sdl2 --disable-bzlib --disable-lzma --disable-linux-perf"
ENABLE="--enable-shared --enable-gpl --enable-version3 --enable-nonfree --enable-runtime-cpudetect --enable-zlib --enable-libmp3lame --enable-libspeex --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libvo-amrwbenc --enable-openssl --enable-libopenh264 --enable-libx264 --enable-libx265 --enable-libvpx --enable-libfreetype --enable-libopus"

# minimal configuration to support MPEG-4 streams with H.264 and AAC as well as Motion JPEG
# DISABLE="--disable-iconv --disable-libxcb --disable-opencl --disable-sdl2 --disable-bzlib --disable-lzma --disable-linux-perf --disable-everything"
# ENABLE="--enable-shared --enable-runtime-cpudetect --enable-libopenh264 --enable-encoder=libopenh264 --enable-encoder=aac --enable-encoder=mjpeg --enable-decoder=h264 --enable-decoder=aac --enable-decoder=mjpeg --enable-parser=h264 --enable-parser=aac --enable-parser=mjpeg --enable-muxer=mp4 --enable-muxer=rtsp --enable-muxer=mjpeg --enable-demuxer=mov --enable-demuxer=rtsp --enable-demuxer=mjpeg --enable-protocol=file --enable-protocol=http --enable-protocol=rtp --enable-protocol=rtmp"

NASM_VERSION=2.13.03
ZLIB=zlib-1.2.11
LAME=lame-3.100
SPEEX=speex-1.2.0
OPUS=opus-1.2.1
OPENCORE_AMR=opencore-amr-0.1.5
VO_AMRWBENC=vo-amrwbenc-0.1.3
OPENSSL=openssl-1.1.0f # openssl-1.1.0g doesn't work on Windows
OPENH264_VERSION=1.7.0
X265=2.6
VPX_VERSION=1.6.1
ALSA_VERSION=1.1.5
FREETYPE_VERSION=2.8.1
MFX_VERSION=1.23
FFMPEG_VERSION=3.4.2
download http://www.nasm.us/pub/nasm/releasebuilds/$NASM_VERSION/nasm-$NASM_VERSION.tar.gz nasm-$NASM_VERSION.tar.gz
download http://zlib.net/$ZLIB.tar.gz $ZLIB.tar.gz
download http://downloads.sourceforge.net/project/lame/lame/3.100/$LAME.tar.gz $LAME.tar.gz
download http://downloads.xiph.org/releases/speex/$SPEEX.tar.gz $SPEEX.tar.gz
download https://archive.mozilla.org/pub/opus/$OPUS.tar.gz $OPUS.tar.gz
download http://sourceforge.net/projects/opencore-amr/files/opencore-amr/$OPENCORE_AMR.tar.gz/download $OPENCORE_AMR.tar.gz
download http://sourceforge.net/projects/opencore-amr/files/vo-amrwbenc/$VO_AMRWBENC.tar.gz/download $VO_AMRWBENC.tar.gz
download https://www.openssl.org/source/$OPENSSL.tar.gz $OPENSSL.tar.gz
download https://github.com/cisco/openh264/archive/v$OPENH264_VERSION.tar.gz openh264-$OPENH264_VERSION.tar.gz
download https://download.videolan.org/x264/snapshots/last_stable_x264.tar.bz2 last_stable_x264.tar.bz2
download https://github.com/videolan/x265/archive/$X265.tar.gz x265-$X265.tar.gz
download https://github.com/webmproject/libvpx/archive/v$VPX_VERSION.tar.gz libvpx-$VPX_VERSION.tar.gz
download https://ftp.osuosl.org/pub/blfs/conglomeration/alsa-lib/alsa-lib-$ALSA_VERSION.tar.bz2 alsa-lib-$ALSA_VERSION.tar.bz2
download http://download.savannah.gnu.org/releases/freetype/freetype-$FREETYPE_VERSION.tar.bz2 freetype-$FREETYPE_VERSION.tar.bz2
download https://github.com/lu-zero/mfx_dispatch/archive/$MFX_VERSION.tar.gz mfx_dispatch-$MFX_VERSION.tar.gz
download http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 ffmpeg-$FFMPEG_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../nasm-$NASM_VERSION.tar.gz
tar --totals -xzf ../$ZLIB.tar.gz
tar --totals -xzf ../$LAME.tar.gz
tar --totals -xzf ../$SPEEX.tar.gz
tar --totals -xzf ../$OPUS.tar.gz
tar --totals -xzf ../$OPENCORE_AMR.tar.gz
tar --totals -xzf ../$VO_AMRWBENC.tar.gz
tar --totals -xzf ../$OPENSSL.tar.gz
tar --totals -xzf ../openh264-$OPENH264_VERSION.tar.gz
tar --totals -xjf ../last_stable_x264.tar.bz2
tar --totals -xzf ../x265-$X265.tar.gz
tar --totals -xzf ../libvpx-$VPX_VERSION.tar.gz
tar --totals -xjf ../freetype-$FREETYPE_VERSION.tar.bz2
tar --totals -xzf ../mfx_dispatch-$MFX_VERSION.tar.gz
tar --totals -xjf ../ffmpeg-$FFMPEG_VERSION.tar.bz2
X264=`echo x264-snapshot-*`

cd nasm-$NASM_VERSION
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ V=0
make install
export PATH=$INSTALL_PATH/bin:$PATH
cd ..

patch -p0 < ../../lame.patch

case $PLATFORM in
    android-arm)
#        ANDROID_ROOT=${ANDROID_ROOT//14/21}
#        ANDROID_FLAGS=${ANDROID_FLAGS//14/21}
        export AR="$ANDROID_BIN-ar"
        export CPP="$ANDROID_BIN-cpp"
        export CC="$ANDROID_BIN-gcc"
        export CXX="$ANDROID_BIN-g++"
        export RANLIB="$ANDROID_BIN-ranlib"
        export STRIP="$ANDROID_BIN-strip"
        export CPPFLAGS="$ANDROID_FLAGS -D_FILE_OFFSET_BITS=32"
        export CFLAGS="$ANDROID_FLAGS -D_FILE_OFFSET_BITS=32"
        export CXXFLAGS="$ANDROID_FLAGS -D_FILE_OFFSET_BITS=32"
        export LDFLAGS="-Wl,--no-undefined -Wl,--fix-cortex-a8 -z text"
        export LIBS="-lgcc -ldl -lz -lm -lc"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=arm-linux
        make -j $MAKEJ V=0
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux
        make -j $MAKEJ V=0
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux
        cd libspeex
        make -j $MAKEJ V=0
        make install
        cd ../include
        make install
        cd ../../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        ./Configure --prefix=$INSTALL_PATH android-armeabi "$CFLAGS" no-shared
        ANDROID_DEV="$ANDROID_ROOT/usr" make -j $MAKEJ
        make install_sw
        cd ../openh264-$OPENH264_VERSION
        LDFLAGS= make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=arm USE_ASM=No NDKROOT="$ANDROID_NDK" TARGET="$ANDROID_ROOT" install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=arm-linux --extra-cflags="$CFLAGS" --extra-ldflags="$LDFLAGS $LIBS"
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        patch -Np1 < ../../../x265-android.patch || true
        $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_TOOLCHAIN_FILE=android-arm.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ x265-static
        make install
        cd ../libvpx-$VPX_VERSION
        patch -Np1 < ../../../libvpx-android.patch
        LDFLAGS= ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --sdk-path=$ANDROID_NDK --disable-tools --target=armv7-android-gcc --disable-runtime-cpu-detect --disable-neon --disable-neon-asm
        make -j $MAKEJ
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic --host=arm-linux
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-android.patch
        sed -i="" 's/_FILE_OFFSET_BITS=64/_FILE_OFFSET_BITS=32/g' configure
        ./configure --prefix=.. $DISABLE $ENABLE --enable-pthreads --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --ranlib="$ANDROID_BIN-ranlib" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=arm --extra-cflags="-I../include/ $CFLAGS" --extra-ldflags="-L../lib/ -L$ANDROID_CPP/libs/armeabi/ $LDFLAGS" --extra-libs="-lgnustl_static $LIBS" --disable-symver --disable-programs
        make -j $MAKEJ
        make install
        ;;

    android-arm64)
        export AR="$ANDROID_BIN-ar"
        export CPP="$ANDROID_BIN-cpp"
        export CC="$ANDROID_BIN-gcc"
        export CXX="$ANDROID_BIN-g++"
        export RANLIB="$ANDROID_BIN-ranlib"
        export STRIP="$ANDROID_BIN-strip"
        export CPPFLAGS="$ANDROID_FLAGS"
        export CFLAGS="$ANDROID_FLAGS"
        export CXXFLAGS="$ANDROID_FLAGS"
        export LDFLAGS="-Wl,--no-undefined -z text"
        export LIBS="-lgcc -ldl -lz -lm -lc"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=aarch64-linux
        make -j $MAKEJ V=0
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux
        make -j $MAKEJ V=0
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux
        cd libspeex
        make -j $MAKEJ V=0
        make install
        cd ../include
        make install
        cd ../../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        ./Configure --prefix=$INSTALL_PATH android64-aarch64 "$CFLAGS" no-shared
        ANDROID_DEV="$ANDROID_ROOT/usr" make -j $MAKEJ
        make install_sw
        cd ../openh264-$OPENH264_VERSION
        LDFLAGS= make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=arm64 USE_ASM=No NDKROOT="$ANDROID_NDK" TARGET="$ANDROID_ROOT" install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=aarch64-linux --extra-cflags="$CFLAGS" --extra-ldflags="$LDFLAGS $LIBS"
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        patch -Np1 < ../../../x265-android.patch || true
        $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_TOOLCHAIN_FILE=android-arm64.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ x265-static
        make install
        cd ../libvpx-$VPX_VERSION
        patch -Np1 < ../../../libvpx-android.patch
        CFLAGS="$CFLAGS -D__uint128_t=__u64" LDFLAGS= ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --sdk-path=$ANDROID_NDK --disable-tools --target=arm64-android-gcc --disable-runtime-cpu-detect --disable-neon --disable-neon-asm
        make -j $MAKEJ
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic --host=aarch64-linux
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-android.patch
        ./configure --prefix=.. $DISABLE $ENABLE --enable-pthreads --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --ranlib="$ANDROID_BIN-ranlib" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=aarch64 --extra-cflags="-I../include/ $CFLAGS" --extra-ldflags="-L../lib/ -L$ANDROID_CPP/libs/arm64-v8a/ $LDFLAGS" --extra-libs="-lgnustl_static $LIBS" --disable-symver --disable-programs
        make -j $MAKEJ
        make install
        ;;

     android-x86)
#        ANDROID_ROOT=${ANDROID_ROOT//14/21}
#        ANDROID_FLAGS=${ANDROID_FLAGS//14/21}
        export AR="$ANDROID_BIN-ar"
        export CPP="$ANDROID_BIN-cpp"
        export CC="$ANDROID_BIN-gcc"
        export CXX="$ANDROID_BIN-g++"
        export RANLIB="$ANDROID_BIN-ranlib"
        export STRIP="$ANDROID_BIN-strip"
        export CPPFLAGS="$ANDROID_FLAGS -D_FILE_OFFSET_BITS=32"
        export CFLAGS="$ANDROID_FLAGS -D_FILE_OFFSET_BITS=32"
        export CXXFLAGS="$ANDROID_FLAGS -D_FILE_OFFSET_BITS=32"
        export LDFLAGS="-Wl,--no-undefined -z text"
        export LIBS="-lgcc -ldl -lz -lm -lc"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=i686-linux
        make -j $MAKEJ V=0
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ V=0
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        cd libspeex
        make -j $MAKEJ V=0
        make install
        cd ../include
        make install
        cd ../../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        ./Configure --prefix=$INSTALL_PATH android-x86 "$CFLAGS" no-shared
        ANDROID_DEV="$ANDROID_ROOT/usr" make -j $MAKEJ
        make install_sw
        cd ../openh264-$OPENH264_VERSION
        LDFLAGS= make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=x86 USE_ASM=No NDKROOT="$ANDROID_NDK" TARGET="$ANDROID_ROOT" install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=i686-linux --disable-asm --extra-cflags="$CFLAGS" --extra-ldflags="$LDFLAGS $LIBS"
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        patch -Np1 < ../../../x265-android.patch || true
        $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_TOOLCHAIN_FILE=android-x86.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ x265-static
        make install
        cd ../libvpx-$VPX_VERSION
        patch -Np1 < ../../../libvpx-android.patch
        ASFLAGS="-D__ANDROID__" LDFLAGS= ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --sdk-path=$ANDROID_NDK --disable-tools --target=x86-android-gcc --as=yasm
        make -j $MAKEJ
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-android.patch
        sed -i="" 's/_FILE_OFFSET_BITS=64/_FILE_OFFSET_BITS=32/g' configure
        ./configure --prefix=.. $DISABLE $ENABLE --enable-pthreads --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --ranlib="$ANDROID_BIN-ranlib" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=atom --extra-cflags="-I../include/ $CFLAGS" --extra-ldflags="-L../lib/ -L$ANDROID_CPP/libs/x86/ $LDFLAGS" --extra-libs="-lgnustl_static $LIBS" --disable-symver --disable-programs
        make -j $MAKEJ
        make install
        ;;

     android-x86_64)
        export AR="$ANDROID_BIN-ar"
        export CPP="$ANDROID_BIN-cpp"
        export CC="$ANDROID_BIN-gcc"
        export CXX="$ANDROID_BIN-g++"
        export RANLIB="$ANDROID_BIN-ranlib"
        export STRIP="$ANDROID_BIN-strip"
        export CPPFLAGS="$ANDROID_FLAGS"
        export CFLAGS="$ANDROID_FLAGS"
        export CXXFLAGS="$ANDROID_FLAGS"
        export LDFLAGS="-Wl,--no-undefined -z text"
        export LIBS="-lgcc -ldl -lz -lm -lc"
        cd $ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=x86_64-linux
        make -j $MAKEJ V=0
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ V=0
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        cd libspeex
        make -j $MAKEJ V=0
        make install
        cd ../include
        make install
        cd ../../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        ./Configure --prefix=$INSTALL_PATH android64 "$CFLAGS" no-shared
        ANDROID_DEV="$ANDROID_ROOT/usr" make -j $MAKEJ
        make install_sw
        cd ../openh264-$OPENH264_VERSION
        LDFLAGS= make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=x86_64 USE_ASM=No NDKROOT="$ANDROID_NDK" TARGET="$ANDROID_ROOT" install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=x86_64-linux --disable-asm --extra-cflags="$CFLAGS" --extra-ldflags="$LDFLAGS $LIBS"
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        patch -Np1 < ../../../x265-android.patch || true
        $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_TOOLCHAIN_FILE=android-x86_64.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ x265-static
        make install
        cd ../libvpx-$VPX_VERSION
        patch -Np1 < ../../../libvpx-android.patch
        ASFLAGS="-D__ANDROID__" LDFLAGS= ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --sdk-path=$ANDROID_NDK --disable-tools --target=x86_64-android-gcc --as=yasm
        make -j $MAKEJ
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-android.patch
        ./configure --prefix=.. $DISABLE $ENABLE --enable-pthreads --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --ranlib="$ANDROID_BIN-ranlib" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=atom --extra-cflags="-I../include/ $CFLAGS" --extra-ldflags="-L../lib/ -L$ANDROID_CPP/libs/x86_64/ $LDFLAGS" --extra-libs="-lgnustl_static $LIBS" --disable-symver --disable-programs
        make -j $MAKEJ
        make install
        ;;

    linux-x86)
        cd $ZLIB
        CC="gcc -m32 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ V=0
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32 -msse2"
        make -j $MAKEJ V=0
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32"
        make -j $MAKEJ V=0
        make install
        cd ../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32"
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32" CXXFLAGS="-m32"
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32" CXXFLAGS="-m32"
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        ./Configure linux-elf -m32 -fPIC no-shared --prefix=$INSTALL_PATH
        make -j $MAKEJ
        make install_sw
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86 USE_ASM=No install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=i686-linux
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=x86-linux-gcc --as=yasm
        make -j $MAKEJ
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic --host=i686-linux CFLAGS="-m32"
        make -j $MAKEJ
        make install
        if [[ ! -z $(ldconfig -p | grep libva-drm) ]]; then
            cd ../mfx_dispatch-$MFX_VERSION
            autoreconf -fiv
            PKG_CONFIG_PATH="../lib/pkgconfig" ./configure --prefix=$INSTALL_PATH --disable-shared --enable-static --enable-fast-install --with-pic --host=i686-linux CFLAGS="-m32" CXXFLAGS="-m32"
            make -j $MAKEJ
            make install
            ENABLE="$ENABLE --enable-libmfx"
        fi
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-linux.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-cuda --enable-cuvid --enable-nvenc --enable-pthreads --enable-libxcb --cc="gcc -m32" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl"
        make -j $MAKEJ
        make install
        ;;

    linux-x86_64)
        cd $ZLIB
        CC="gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ V=0
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        cd ../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64" CXXFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64" CXXFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        ./Configure linux-x86_64 -fPIC no-shared --prefix=$INSTALL_PATH
        make -j $MAKEJ
        make install_sw
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86_64 USE_ASM=No install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=x86_64-linux
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=x86_64-linux-gcc --as=yasm
        make -j $MAKEJ
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
        make -j $MAKEJ
        make install
        if [[ ! -z $(ldconfig -p | grep libva-drm) ]]; then
            cd ../mfx_dispatch-$MFX_VERSION
            autoreconf -fiv
            PKG_CONFIG_PATH="../lib/pkgconfig" ./configure --prefix=$INSTALL_PATH --disable-shared --enable-static --enable-fast-install --with-pic --host=x86_64-linux CFLAGS="-m64" CXXFLAGS="-m64"
            make -j $MAKEJ
            make install
            ENABLE="$ENABLE --enable-libmfx"
        fi
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-linux.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-cuda --enable-cuvid --enable-nvenc --enable-pthreads --enable-libxcb --cc="gcc -m64" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl"
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
        make -j $MAKEJ V=0
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ V=0
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ V=0
        make install
        cd ../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        if [ $CROSSCOMPILE -eq 1 ]
        then
          ./Configure linux-generic32 -march=armv6 -mfpu=vfp -mfloat-abi=hard -fPIC no-shared --prefix=$INSTALL_PATH --cross-compile-prefix=arm-linux-gnueabihf-
        else
          ./Configure linux-generic32 -fPIC no-shared --prefix=$INSTALL_PATH
        fi
        make -j $MAKEJ
        make install_sw
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=arm-linux-gnueabihf-ar ARCH=armhf USE_ASM=No install-static CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++
        cd ../$X264
        if [ $CROSSCOMPILE -eq 1 ]
        then
          ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --disable-cli --host=arm-linux-gnueabihf --cross-prefix="arm-linux-gnueabihf-" --disable-asm
        else
          ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --disable-cli --disable-asm
        fi
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        if [ $CROSSCOMPILE -eq 1 ]
        then
          $CMAKE -DENABLE_ASSEMBLY=OFF -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_STRIP=arm-linux-gnueabihf-strip -DCMAKE_FIND_ROOT_PATH=arm-linux-gnueabihf -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        else
          $CMAKE -DENABLE_ASSEMBLY=OFF -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        fi
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        if [ $CROSSCOMPILE -eq 1 ]
        then
          CROSS=arm-linux-gnueabihf- ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=armv7-linux-gcc
        else
          ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests
        fi
        make -j $MAKEJ
        make install
        cd ../alsa-lib-$ALSA_VERSION/
        ./configure --host=arm-linux-gnueabihf --prefix=$INSTALL_PATH --disable-python
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-linux.patch
        if [ $CROSSCOMPILE -eq 1 ]
        then
          USERLAND_PATH="$(which arm-linux-gnueabihf-gcc | grep -o '.*/tools/')../userland"
          PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-omx --enable-mmal --enable-omx-rpi --enable-pthreads --cc="arm-linux-gnueabihf-gcc" --extra-cflags="-I$USERLAND_PATH/ -I$USERLAND_PATH/interface/vmcs_host/khronos/IL/ -I$USERLAND_PATH/host_applications/linux/libs/bcm_host/include/ -I../include/" --extra-ldflags="-L$USERLAND_PATH/build/lib/ -L../lib/" --extra-libs="-lstdc++ -ldl -lasound -lvcos" --enable-cross-compile --arch=armhf --target-os=linux --cross-prefix="arm-linux-gnueabihf-" --pkg-config-flags="--static" --pkg-config="pkg-config --static" --disable-doc --disable-programs
        else
          PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-omx --enable-mmal --enable-omx-rpi --enable-pthreads --extra-cflags="-I../include/" --extra-ldflags="-L../lib/ -L/opt/vc/lib" --extra-libs="-lstdc++ -ldl -lasound -lvcos" --pkg-config-flags="--static" --pkg-config="pkg-config --static"
        fi
        make -j $MAKEJ
        make install
        ;;

    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        cd $ZLIB
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        else
          CC="powerpc64le-linux-gnu-gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        fi
        make -j $MAKEJ V=0
        make install
        cd ../$LAME
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64"
        else
          CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" ./configure --host=powerpc64le-linux-gnu --prefix=$INSTALL_PATH --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64"
        fi
        make -j $MAKEJ V=0
        make install
        cd ../$SPEEX
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64"
        else
          ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=powerpc64le-linux-gnu --build=ppc64le-linux CFLAGS="-m64"
        fi
        make -j $MAKEJ V=0
        make install
        cd ../$OPUS
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64" CXXFLAGS="-m64"
        else
          ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=powerpc64le-linux-gnu --build=ppc64le-linux CFLAGS="-m64"
        fi
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64" CXXFLAGS="-m64"
        else
          ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=powerpc64le-linux-gnu --build=ppc64le-linux CFLAGS="-m64"
        fi
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64" CXXFLAGS="-m64"
        else
          ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=powerpc64le-linux-gnu --build=ppc64le-linux CFLAGS="-m64"
        fi
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./Configure linux-ppc64le -fPIC no-shared --prefix=$INSTALL_PATH
        else
          ./Configure linux-ppc64le -fPIC no-shared --cross-compile-prefix=powerpc64le-linux-gnu- --prefix=$INSTALL_PATH
        fi
        make -j $MAKEJ
        make install_sw
        cd ../openh264-$OPENH264_VERSION
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=ppc64le USE_ASM=No install-static
        else
          make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=powerpc64le-linux-gnu-ar ARCH=ppc64le USE_ASM=No install-static CC=powerpc64le-linux-gnu-gcc CXX=powerpc64le-linux-gnu-g++
        fi
        cd ../$X264
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=ppc64le-linux
        else
          CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --build=ppc64le-linux --host=ppc64le-linux
        fi
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64" CXX="g++ -m64" $CMAKE -DENABLE_ASSEMBLY=OFF -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        else
          $CMAKE -DENABLE_ASSEMBLY=OFF -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=ppc64le -DCMAKE_CXX_FLAGS="-m64" -DCMAKE_C_FLAGS="-m64" -DCMAKE_C_COMPILER=powerpc64le-linux-gnu-gcc -DCMAKE_CXX_COMPILER=powerpc64le-linux-gnu-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_STRIP=powerpc64le-linux-gnu-strip -DCMAKE_FIND_ROOT_PATH=powerpc64le-linux-gnu -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        fi
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=generic-gnu
        else
          CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=generic-gnu
        fi
        make -j $MAKEJ
        make install
        cd ../freetype-$FREETYPE_VERSION
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic --target=ppc64le-linux CFLAGS="-m64"
        else
          CC="powerpc64le-linux-gnu-gcc" CXX="powerpc64le-linux-gnu-g++" ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic --host=powerpc64le-linux-gnu --build=ppc64le-linux CFLAGS="-m64"
        fi
        make -j $MAKEJ
        make install 
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-linux.patch
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-pthreads --enable-libxcb --cc="gcc -m64" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl"
        else
          echo "configure ffmpeg cross compile"
          PKG_CONFIG_PATH=../lib/pkgconfig/:/usr/lib/powerpc64le-linux-gnu/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-pthreads --enable-libxcb --cc="powerpc64le-linux-gnu-gcc -m64" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --enable-cross-compile --target-os=linux --arch=ppc64le-linux --extra-libs="-lstdc++ -ldl"
        fi
        make -j $MAKEJ
        make install
        ;;

    macosx-*)
        cd $ZLIB
        CC="clang -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ V=0
        make install
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ V=0
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ V=0
        make install
        cd ../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        ./Configure darwin64-x86_64-cc -fPIC no-shared --prefix=$INSTALL_PATH
        make -j $MAKEJ
        make install_sw
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar USE_ASM=No install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        CC="clang" CXX="clang++" $CMAKE -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests
        make -j $MAKEJ
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-macosx.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-pthreads --enable-indev=avfoundation --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl" --disable-doc --disable-programs
        make -j $MAKEJ
        make install
        ;;

    windows-x86)
        cd $ZLIB
        make -j $MAKEJ install -fwin32/Makefile.gcc BINARY_PATH=$INSTALL_PATH/bin/ INCLUDE_PATH=$INSTALL_PATH/include/ LIBRARY_PATH=$INSTALL_PATH/lib/
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 CFLAGS="-m32 -msse2"
        make -j $MAKEJ V=0
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 CFLAGS="-m32"
        make -j $MAKEJ V=0
        make install
        cd ../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 CFLAGS="-m32"
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 CFLAGS="-m32" CXXFLAGS="-m32"
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 CFLAGS="-m32" CXXFLAGS="-m32"
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        ./Configure mingw -fPIC no-shared --prefix=$INSTALL_PATH
        make -j $MAKEJ
        make install_sw
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86 USE_ASM=No install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=i686-w64-mingw32
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -G "MSYS Makefiles" -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=x86-win32-gcc
        make -j $MAKEJ
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic --host=i686-w64-mingw32 CFLAGS="-m32"
        make -j $MAKEJ
        make install
        cd ../mfx_dispatch-$MFX_VERSION
        autoreconf -fiv
        PKG_CONFIG_PATH="../lib/pkgconfig" ./configure --prefix=$INSTALL_PATH --disable-shared --enable-static --enable-fast-install --with-pic --host=i686-w64-mingw32 # CFLAGS="-m32" CXXFLAGS="-m32"
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-windows.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-cuda --enable-cuvid --enable-nvenc --enable-libmfx --enable-w32threads --enable-indev=dshow --target-os=mingw32 --cc="gcc -m32" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lgcc -lgcc_eh -lWs2_32 -lcrypt32 -lpthread -Wl,-Bdynamic"
        make -j $MAKEJ
        make install
        ;;

    windows-x86_64)
        cd $ZLIB
        make -j $MAKEJ install -fwin32/Makefile.gcc BINARY_PATH=$INSTALL_PATH/bin/ INCLUDE_PATH=$INSTALL_PATH/include/ LIBRARY_PATH=$INSTALL_PATH/lib/
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 CFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 CFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        cd ../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 CFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 CFLAGS="-m64" CXXFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 CFLAGS="-m64" CXXFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        ./Configure mingw64 -fPIC no-shared --prefix=$INSTALL_PATH
        make -j $MAKEJ
        make install_sw
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86_64 USE_ASM=No install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=x86_64-w64-mingw32
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -G "MSYS Makefiles" -DENABLE_SHARED=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=x86_64-win64-gcc
        make -j $MAKEJ
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --enable-static --disable-shared --with-pic --host=x86_64-w64-mingw32 CFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../mfx_dispatch-$MFX_VERSION
        autoreconf -fiv
        PKG_CONFIG_PATH="../lib/pkgconfig" ./configure --prefix=$INSTALL_PATH --disable-shared --enable-static --enable-fast-install --with-pic --host=x86_64-w64-mingw32 # CFLAGS="-m64" CXXFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-windows.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-cuda --enable-cuvid --enable-nvenc --enable-libmfx --enable-w32threads --enable-indev=dshow --target-os=mingw32 --cc="gcc -m64" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lgcc -lgcc_eh -lWs2_32 -lcrypt32 -lpthread -Wl,-Bdynamic"
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..

