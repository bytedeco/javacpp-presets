#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" ffmpeg
    popd
    exit
fi

DISABLE="--disable-w32threads --disable-iconv --disable-libxcb --disable-opencl --disable-sdl"
ENABLE="--enable-pthreads --enable-shared --enable-gpl --enable-version3 --enable-nonfree --enable-runtime-cpudetect --enable-libmp3lame --enable-libspeex --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-openssl --enable-libopenh264 --enable-libx264 --enable-libx265 --enable-libvpx"

# minimal configuration to support MPEG-4 streams with H.264 and AAC
# DISABLE="--disable-w32threads --disable-iconv --disable-libxcb --disable-opencl --disable-sdl --disable-zlib --disable-everything"
# ENABLE="--enable-pthreads --enable-shared --enable-runtime-cpudetect --enable-libopenh264 --enable-encoder=libopenh264 --enable-encoder=aac --enable-decoder=h264 --enable-decoder=aac --enable-parser=h264 --enable-parser=aac --enable-muxer=mp4 --enable-muxer=rtsp --enable-demuxer=mov --enable-demuxer=rtsp --enable-protocol=file --enable-protocol=http --enable-protocol=rtp --enable-protocol=rtmp"

if [[ $PLATFORM == windows* && !($DISABLE =~ "--disable-everything") ]]; then
    FFMPEG_VERSION=2.8.4
    [[ $PLATFORM == *64 ]] && BITS=64 || BITS=32
    download http://ffmpeg.zeranoe.com/builds/win$BITS/dev/ffmpeg-$FFMPEG_VERSION-win$BITS-dev.7z ffmpeg-$FFMPEG_VERSION-win$BITS-dev.7z
    download http://ffmpeg.zeranoe.com/builds/win$BITS/shared/ffmpeg-$FFMPEG_VERSION-win$BITS-shared.7z ffmpeg-$FFMPEG_VERSION-win$BITS-shared.7z

    mkdir -p $PLATFORM
    cd $PLATFORM
    7z x -y ../ffmpeg-$FFMPEG_VERSION-win$BITS-dev.7z
    7z x -y ../ffmpeg-$FFMPEG_VERSION-win$BITS-shared.7z
else
    LAME=lame-3.99.5
    SPEEX=speex-1.2rc2
    OPENCORE_AMR=opencore-amr-0.1.3
    OPENSSL=openssl-1.0.2e
    OPENH264_VERSION=1.5.0
    X265=x265_1.8
    VPX_VERSION=v1.5.0
    FFMPEG_VERSION=2.8.4
    download http://downloads.sourceforge.net/project/lame/lame/3.99/$LAME.tar.gz $LAME.tar.gz
    download http://downloads.xiph.org/releases/speex/$SPEEX.tar.gz $SPEEX.tar.gz
    download http://sourceforge.net/projects/opencore-amr/files/opencore-amr/$OPENCORE_AMR.tar.gz/download $OPENCORE_AMR.tar.gz
    download https://www.openssl.org/source/$OPENSSL.tar.gz $OPENSSL.tar.gz
    download https://github.com/cisco/openh264/archive/v$OPENH264_VERSION.tar.gz openh264-$OPENH264_VERSION.tar.gz
    download ftp://ftp.videolan.org/pub/videolan/x264/snapshots/last_stable_x264.tar.bz2 last_stable_x264.tar.bz2
    download https://ftp.videolan.org/pub/videolan/x265/$X265.tar.gz $X265.tar.gz
    download https://chromium.googlesource.com/webm/libvpx/+archive/$VPX_VERSION.tar.gz libvpx-$VPX_VERSION.tar.gz
    download http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 ffmpeg-$FFMPEG_VERSION.tar.bz2

    mkdir -p $PLATFORM
    cd $PLATFORM
    INSTALL_PATH=`pwd`
    tar -xzvf ../$LAME.tar.gz
    tar -xzvf ../$SPEEX.tar.gz
    tar -xzvf ../$OPENCORE_AMR.tar.gz
    tar -xzvf ../$OPENSSL.tar.gz
    tar -xzvf ../openh264-$OPENH264_VERSION.tar.gz
    tar -xjvf ../last_stable_x264.tar.bz2
    tar -xzvf ../$X265.tar.gz
    mkdir -p libvpx-$VPX_VERSION
    tar -xzvf ../libvpx-$VPX_VERSION.tar.gz -C libvpx-$VPX_VERSION
    tar -xjvf ../ffmpeg-$FFMPEG_VERSION.tar.bz2
    X264=`echo x264-snapshot-*`
fi

case $PLATFORM in
    android-arm)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux CPP="$ANDROID_BIN-cpp" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j $MAKEJ
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux CPP="$ANDROID_BIN-cpp" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        cd libspeex
        make -j $MAKEJ
        make install
        cd ../include
        make install
        cd ../../$OPENCORE_AMR
        BUILD_FLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux CPP="$ANDROID_BIN-cpp" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" RANLIB="$ANDROID_BIN-ranlib" CFLAGS="$BUILD_FLAGS" CXXFLAGS="$BUILD_FLAGS" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j $MAKEJ
        make install
        cd ../$OPENSSL
        CROSS_COMPILE="$ANDROID_BIN-" ./Configure android-armv7 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300 no-shared --prefix=$INSTALL_PATH
        ANDROID_DEV="$ANDROID_ROOT/usr" make # fails with -j > 1
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=arm USE_ASM=No NDKROOT="$ANDROID_NDK" TARGET="$ANDROID_ROOT" libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=arm-linux --extra-cflags="-DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="-nostdlib -Wl,--fix-cortex-a8 -lgcc -ldl -lz -lm -lc"
        make -j $MAKEJ
        make install
        cd ../$X265
        patch -Np1 < ../../../$X265-android.patch || true
        $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DCMAKE_TOOLCHAIN_FILE=android-arm.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --sdk-path=$ANDROID_NDK --target=armv7-android-gcc --disable-runtime-cpu-detect --disable-neon --disable-neon-asm
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-android.patch
        ./configure --prefix=.. $DISABLE $ENABLE --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --ranlib="$ANDROID_BIN-ranlib" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=arm --extra-cflags="-I../include/ -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="$ANDROID_ROOT/usr/lib/crtbegin_so.o -L../lib/ -L$ANDROID_CPP/libs/armeabi/ -nostdlib -Wl,--fix-cortex-a8" --extra-libs="-lgnustl_static -lgcc -ldl -lz -lm -lc" --disable-symver --disable-programs
        make -j $MAKEJ
        make install
        ;;

     android-x86)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CPP="$ANDROID_BIN-cpp" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j $MAKEJ
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CPP="$ANDROID_BIN-cpp" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        cd libspeex
        make -j $MAKEJ
        make install
        cd ../include
        make install
        cd ../../$OPENCORE_AMR
        BUILD_FLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CPP="$ANDROID_BIN-cpp" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" RANLIB="$ANDROID_BIN-ranlib" CFLAGS="$BUILD_FLAGS" CXXFLAGS="$BUILD_FLAGS" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j $MAKEJ
        make install
        cd ../$OPENSSL
        CROSS_COMPILE="$ANDROID_BIN-" ./Configure android-x86 -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300 no-shared --prefix=$INSTALL_PATH
        ANDROID_DEV="$ANDROID_ROOT/usr" make # fails with -j > 1
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=x86 NDKROOT="$ANDROID_NDK" TARGET="$ANDROID_ROOT" libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=i686-linux --extra-cflags="-DANDROID -fPIC -ffunction-sections -funwind-tables -mtune=atom -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="-nostdlib -lgcc -ldl -lz -lm -lc"
        make -j $MAKEJ
        make install
        cd ../$X265
        patch -Np1 < ../../../$X265-android.patch || true
        $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DCMAKE_TOOLCHAIN_FILE=android-x86.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ x265-static
        make install
        cd ../libvpx-$VPX_VERSION
        CROSS="$ANDROID_BIN-" ASFLAGS="-D__ANDROID__" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mtune=atom -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="--sysroot=$ANDROID_ROOT -nostdlib -lgcc -ldl -lz -lm -lc" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --target=x86-android-gcc
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-android.patch
        ./configure --prefix=.. $DISABLE $ENABLE --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --ranlib="$ANDROID_BIN-ranlib" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=atom --extra-cflags="-I../include/ -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="$ANDROID_ROOT/usr/lib/crtbegin_so.o -L../lib/ -L$ANDROID_CPP/libs/x86/ -nostdlib" --extra-libs="-lgnustl_static -lgcc -ldl -lz -lm -lc" --disable-symver --disable-programs
        make -j $MAKEJ
        make install
        ;;

    linux-x86)
        cd $LAME
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
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86 libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../$X265
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --target=x86-linux-gcc
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        [[ $ENABLE =~ "--enable-gpl" ]] && X11GRAB="--enable-x11grab" || X11GRAB=
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $X11GRAB --cc="gcc -m32" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl"
        make -j $MAKEJ
        make install
        ;;

    linux-x86_64)
        cd $LAME
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
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86_64 libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=x86_64-linux
        make -j $MAKEJ
        make install
        cd ../$X265
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j $MAKEJ
        make install
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --target=x86_64-linux-gcc
        make -j $MAKEJ
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        [[ $ENABLE =~ "--enable-gpl" ]] && X11GRAB="--enable-x11grab" || X11GRAB=
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $X11GRAB --cc="gcc -m64" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl"
        make -j $MAKEJ
        make install
        ;;

    macosx-*)
        cd $LAME
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
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar libraries install-static
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
        if [[ !($DISABLE =~ "--disable-everything") ]]; then
            cp -r ffmpeg-$FFMPEG_VERSION-win32-dev/include .
            cp -r ffmpeg-$FFMPEG_VERSION-win32-dev/lib .
            cp -r ffmpeg-$FFMPEG_VERSION-win32-shared/bin .
            cd ..
            return
        fi

        cd $LAME
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
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86 libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=i686-w64-mingw32
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
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-indev=dshow --target-os=mingw32 --cc="gcc -m32" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic"
        make -j $MAKEJ
        make install
        ;;

    windows-x86_64)
        if [[ !($DISABLE =~ "--disable-everything") ]]; then
            cp -r ffmpeg-$FFMPEG_VERSION-win64-dev/include .
            cp -r ffmpeg-$FFMPEG_VERSION-win64-dev/lib .
            cp -r ffmpeg-$FFMPEG_VERSION-win64-shared/bin .
            cd ..
            return
        fi

        cd $LAME
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
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86_64 libraries install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=x86_64-w64-mingw32
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
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE --enable-indev=dshow --target-os=mingw32 --cc="gcc -m64" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic"
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
