#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" ffmpeg
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    FFMPEG_VERSION=20150316-git-1e4d049
    [[ $PLATFORM == *64 ]] && BITS=64 || BITS=32
    download http://ffmpeg.zeranoe.com/builds/win$BITS/dev/ffmpeg-$FFMPEG_VERSION-win$BITS-dev.7z ffmpeg-$FFMPEG_VERSION-win$BITS-dev.7z
    download http://ffmpeg.zeranoe.com/builds/win$BITS/shared/ffmpeg-$FFMPEG_VERSION-win$BITS-shared.7z ffmpeg-$FFMPEG_VERSION-win$BITS-shared.7z
    download http://msinttypes.googlecode.com/files/msinttypes-r26.zip msinttypes-r26.zip

    mkdir -p $PLATFORM
    cd $PLATFORM
    7za x -y ../ffmpeg-$FFMPEG_VERSION-win$BITS-dev.7z
    7za x -y ../ffmpeg-$FFMPEG_VERSION-win$BITS-shared.7z
    patch -Np1 -d ffmpeg-$FFMPEG_VERSION-win$BITS-dev/ < ../../ffmpeg-$FFMPEG_VERSION-windows.patch
else
    LAME=lame-3.99.5
    SPEEX=speex-1.2rc1
    OPENCORE_AMR=opencore-amr-0.1.3
    OPENSSL=openssl-1.0.2a
    X265=x265_1.5
    FFMPEG_VERSION=2.6.1
    download http://downloads.sourceforge.net/project/lame/lame/3.99/$LAME.tar.gz $LAME.tar.gz
    download http://downloads.xiph.org/releases/speex/$SPEEX.tar.gz $SPEEX.tar.gz
    download http://sourceforge.net/projects/opencore-amr/files/opencore-amr/$OPENCORE_AMR.tar.gz/download $OPENCORE_AMR.tar.gz
    download https://www.openssl.org/source/$OPENSSL.tar.gz $OPENSSL.tar.gz
    download ftp://ftp.videolan.org/pub/videolan/x264/snapshots/last_stable_x264.tar.bz2 last_stable_x264.tar.bz2
    download https://bitbucket.org/multicoreware/x265/downloads/$X265.tar.gz $X265.tar.gz
    download http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 ffmpeg-$FFMPEG_VERSION.tar.bz2

    mkdir -p $PLATFORM
    cd $PLATFORM
    INSTALL_PATH=`pwd`
    tar -xzvf ../$LAME.tar.gz
    tar -xzvf ../$SPEEX.tar.gz
    tar -xzvf ../$OPENCORE_AMR.tar.gz
    tar -xzvf ../$OPENSSL.tar.gz
    tar -xjvf ../last_stable_x264.tar.bz2
    tar -xzvf ../$X265.tar.gz
    tar -xjvf ../ffmpeg-$FFMPEG_VERSION.tar.bz2
    X264=`echo x264-snapshot-*`
fi

case $PLATFORM in
    android-arm)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        cd libspeex
        make -j4
        make install
        cd ../include
        make install
        cd ../../$OPENCORE_AMR
        BUILD_FLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" RANLIB="$ANDROID_BIN-ranlib" CFLAGS="$BUILD_FLAGS" CXXFLAGS="$BUILD_FLAGS" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install
        cd ../$OPENSSL
        CROSS_COMPILE="$ANDROID_BIN-" ./Configure android-armv7 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300 no-shared --prefix=$INSTALL_PATH
        ANDROID_DEV="$ANDROID_ROOT/usr" make
        make install
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=arm-linux --extra-cflags="-DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="-nostdlib -Wl,--fix-cortex-a8 -lgcc -ldl -lz -lm -lc"
        make -j4
        make install
        cd ../$X265
        patch -Np1 < ../../../$X265-android.patch || true
        $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DCMAKE_TOOLCHAIN_FILE=android-arm.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j4
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-android.patch
        ./configure --prefix=.. --enable-shared --enable-gpl --enable-version3 --enable-nonfree --enable-runtime-cpudetect --disable-iconv --disable-libxcb --disable-opencl --disable-outdev=sdl --enable-libmp3lame --enable-libspeex --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-openssl --enable-libx264 --enable-libx265 --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --ranlib="$ANDROID_BIN-ranlib" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=arm --extra-cflags="-I../include/ -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="$ANDROID_ROOT/usr/lib/crtbegin_so.o -L../lib/ -L$ANDROID_CPP/libs/armeabi/ -nostdlib -Wl,--fix-cortex-a8" --extra-libs="-lgnustl_static -lgcc -ldl -lz -lm -lc" --disable-symver --disable-programs
        make -j4
        make install
        ;;
     android-x86)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        cd libspeex
        make -j4
        make install
        cd ../include
        make install
        cd ../../$OPENCORE_AMR
        BUILD_FLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300"
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" RANLIB="$ANDROID_BIN-ranlib" CFLAGS="$BUILD_FLAGS" CXXFLAGS="$BUILD_FLAGS" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install
        cd ../$OPENSSL
        CROSS_COMPILE="$ANDROID_BIN-" ./Configure android-x86 -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300 no-shared --prefix=$INSTALL_PATH
        ANDROID_DEV="$ANDROID_ROOT/usr" make
        make install
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=i686-linux --extra-cflags="-DANDROID -fPIC -ffunction-sections -funwind-tables -mtune=atom -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="-nostdlib -lgcc -ldl -lz -lm -lc"
        make -j4
        make install
        cd ../$X265
        patch -Np1 < ../../../$X265-android.patch || true
        $CMAKE -DENABLE_CLI=OFF -DENABLE_SHARED=OFF -DCMAKE_TOOLCHAIN_FILE=android-x86.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j4 x265-static
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-android.patch
        ./configure --prefix=.. --enable-shared --enable-gpl --enable-version3 --enable-nonfree --enable-runtime-cpudetect --disable-iconv --disable-libxcb --disable-opencl --disable-outdev=sdl --enable-libmp3lame --enable-libspeex --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-openssl --enable-libx264 --enable-libx265 --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --ranlib="$ANDROID_BIN-ranlib" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=atom --extra-cflags="-I../include/ -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="$ANDROID_ROOT/usr/lib/crtbegin_so.o -L../lib/ -L$ANDROID_CPP/libs/x86/ -nostdlib" --extra-libs="-lgnustl_static -lgcc -ldl -lz -lm -lc" --disable-symver --disable-programs
        make -j4
        make install
        ;;
    linux-x86)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32 -msse2"
        make -j4
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32"
        make -j4
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32" CXXFLAGS="-m32"
        make -j4
        make install
        cd ../$OPENSSL
        ./Configure linux-elf -m32 -fPIC no-shared --prefix=$INSTALL_PATH
        make
        make install
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=i686-linux
        make -j4
        make install
        cd ../$X265
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j4
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. --enable-shared --enable-gpl --enable-version3 --enable-nonfree --enable-runtime-cpudetect --disable-iconv --disable-libxcb --disable-opencl --disable-outdev=sdl --enable-x11grab --enable-libmp3lame --enable-libspeex --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-openssl --enable-libx264 --enable-libx265 --cc="gcc -m32" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl"
        make -j4
        make install
        ;;
    linux-x86_64)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
        make -j4
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
        make -j4
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64" CXXFLAGS="-m64"
        make -j4
        make install
        cd ../$OPENSSL
        ./Configure linux-x86_64 -fPIC no-shared --prefix=$INSTALL_PATH
        make
        make install
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=x86_64-linux
        make -j4
        make install
        cd ../$X265
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j4
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. --enable-shared --enable-gpl --enable-version3 --enable-nonfree --enable-runtime-cpudetect --disable-iconv --disable-libxcb --disable-opencl --disable-outdev=sdl --enable-x11grab --enable-libmp3lame --enable-libspeex --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-openssl --enable-libx264 --enable-libx265 --cc="gcc -m64" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl"
        make -j4
        make install
        ;;
    macosx-*)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j4
        make install
        cd ../$SPEEX
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j4
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j4
        make install
        cd ../$OPENSSL
        ./Configure darwin64-x86_64-cc -fPIC no-shared --prefix=$INSTALL_PATH
        make
        make install
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl
        make -j4
        make install
        cd ../$X265
        CC="clang" CXX="clang++" $CMAKE -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. source
        make -j4
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-macosx.patch
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. --enable-shared --enable-gpl --enable-version3 --enable-nonfree --enable-runtime-cpudetect --disable-iconv --disable-libxcb --disable-opencl --disable-outdev=sdl --enable-indev=avfoundation --enable-libmp3lame --enable-libspeex --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-openssl --enable-libx264 --enable-libx265 --extra-cflags="-I../include/" --extra-ldflags="-L../lib/ -Wl,-headerpad_max_install_names" --extra-libs="-lstdc++ -ldl" --disable-doc
        make -j4
        make install
        ;;
    windows-x86)
        # http://ffmpeg.org/platform.html#Linking-to-FFmpeg-with-Microsoft-Visual-C_002b_002b
        LIBS=(avcodec-56 avdevice-56 avfilter-5 avformat-56 avutil-54 postproc-53 swresample-1 swscale-3)
        for LIB in ${LIBS[@]}; do
            lib /def:ffmpeg-$FFMPEG_VERSION-win32-dev/lib/$LIB.def /out:ffmpeg-$FFMPEG_VERSION-win32-dev/lib/$LIB.lib /machine:x86
        done
        cp -r ffmpeg-$FFMPEG_VERSION-win32-dev/include .
        cp -r ffmpeg-$FFMPEG_VERSION-win32-dev/lib .
        cp -r ffmpeg-$FFMPEG_VERSION-win32-shared/bin .
        cd include
        unzip -o ../../msinttypes-r26.zip
        ;;
    windows-x86_64)
        # http://ffmpeg.org/platform.html#Linking-to-FFmpeg-with-Microsoft-Visual-C_002b_002b
        LIBS=(avcodec-56 avdevice-56 avfilter-5 avformat-56 avutil-54 postproc-53 swresample-1 swscale-3)
        for LIB in ${LIBS[@]}; do
            lib /def:ffmpeg-$FFMPEG_VERSION-win64-dev/lib/$LIB.def /out:ffmpeg-$FFMPEG_VERSION-win64-dev/lib/$LIB.lib /machine:x64
        done
        cp -r ffmpeg-$FFMPEG_VERSION-win64-dev/include .
        cp -r ffmpeg-$FFMPEG_VERSION-win64-dev/lib .
        cp -r ffmpeg-$FFMPEG_VERSION-win64-shared/bin .
        cd include
        unzip -o ../../msinttypes-r26.zip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
