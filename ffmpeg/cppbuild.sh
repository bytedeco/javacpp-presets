#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" ffmpeg
    popd
    exit
fi

X264STATICOPTS="--enable-static --disable-opencl --disable-avs --disable-cli --disable-ffms --disable-gpac --disable-lavf --disable-swscale"

if [[ $PLATFORM == windows* ]]; then
    FFMPEG_VERSION=20140716-git-faafd1e
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
    FFMPEG_VERSION=2.3
    download http://downloads.sourceforge.net/project/lame/lame/3.99/lame-3.99.5.tar.gz lame-3.99.5.tar.gz
    download ftp://ftp.videolan.org/pub/videolan/x264/snapshots/last_stable_x264.tar.bz2 last_stable_x264.tar.bz2
    download http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 ffmpeg-$FFMPEG_VERSION.tar.bz2

    mkdir -p $PLATFORM
    cd $PLATFORM
    INSTALL_PATH=`pwd`
    tar -xzvf ../lame-3.99.5.tar.gz
    tar -xjvf ../last_stable_x264.tar.bz2
    tar -xjvf ../ffmpeg-$FFMPEG_VERSION.tar.bz2
    LAME=lame-3.99.5
    X264=`echo x264-snapshot-*`
fi

case $PLATFORM in
    android-arm)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=arm-linux --extra-cflags="-DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="-nostdlib -Wl,--fix-cortex-a8 -lgcc -ldl -lz -lm -lc"
        make -j4
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-android.patch
        ./configure --prefix=.. --enable-shared --enable-gpl --enable-version3 --enable-runtime-cpudetect --disable-outdev=sdl --enable-libmp3lame --enable-libx264 --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=arm --extra-cflags="-I../include/ -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="-L../lib/ -nostdlib -Wl,--fix-cortex-a8" --extra-libs="-lgcc -ldl -lz -lm -lc" --disable-symver --disable-programs
        make -j4
        make install
        ;;
     android-x86)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=i686-linux --extra-cflags="-DANDROID -fPIC -ffunction-sections -funwind-tables -mtune=atom -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="-nostdlib -lgcc -ldl -lz -lm -lc"
        make -j4
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-$FFMPEG_VERSION-android.patch
        ./configure --prefix=.. --enable-shared --enable-gpl --enable-version3 --enable-runtime-cpudetect --disable-outdev=sdl --enable-libmp3lame --enable-libx264 --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --target-os=linux --arch=atom --extra-cflags="-I../include/ -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" --extra-ldflags="-L../lib/ -nostdlib" --extra-libs="-lgcc -ldl -lz -lm -lc" --disable-symver --disable-programs
        make -j4
        make install
        ;;
    linux-x86)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32"
        make -j4
        make install
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --host=i686-linux
        make -j4
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        ./configure --prefix=.. --enable-shared --enable-gpl --enable-version3 --enable-runtime-cpudetect --disable-outdev=sdl --enable-libmp3lame --enable-libx264 --cc="gcc -m32" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/ -ldl"
        make -j4
        make install
        ;;
    linux-x86_64)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
        make -j4
        make install
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH $X264STATICOPTS --enable-pic --host=x86_64-linux
        make -j4
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        ./configure --prefix=.. --enable-shared --enable-gpl --enable-version3 --enable-runtime-cpudetect --disable-outdev=sdl --enable-libmp3lame --enable-libx264 --cc="gcc -m64" --extra-cflags="-I../include/" --extra-ldflags="-L../lib/ -ldl"
        make -j4
        make install
        ;;
    macosx-*)
        cd $LAME
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j4
        make install
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic
        make -j4
        make install
        cd ../ffmpeg-$FFMPEG_VERSION
        ./configure --prefix=.. --enable-shared --enable-gpl --enable-version3 --enable-runtime-cpudetect --disable-outdev=sdl --enable-libmp3lame --enable-libx264 --extra-cflags="-I../include/" --extra-ldflags="-L../lib/ -Wl,-headerpad_max_install_names -ldl"
        make -j4
        make install
        BADPATH=../lib
        LIBS="libavcodec.55.dylib libavdevice.55.dylib libavfilter.4.dylib libavformat.55.dylib libavutil.52.dylib libpostproc.52.dylib libswresample.0.dylib libswscale.2.dylib"
        for f in $LIBS; do install_name_tool $BADPATH/$f -id @rpath/$f \
            -add_rpath /usr/local/lib/ -add_rpath /opt/local/lib/ -add_rpath @loader_path/. \
            -change $BADPATH/libavcodec.55.dylib @rpath/libavcodec.55.dylib \
            -change $BADPATH/libavdevice.55.dylib @rpath/libavdevice.55.dylib \
            -change $BADPATH/libavfilter.4.dylib @rpath/libavfilter.4.dylib \
            -change $BADPATH/libavformat.55.dylib @rpath/libavformat.55.dylib \
            -change $BADPATH/libavutil.52.dylib @rpath/libavutil.52.dylib \
            -change $BADPATH/libpostproc.52.dylib @rpath/libpostproc.52.dylib \
            -change $BADPATH/libswresample.0.dylib @rpath/libswresample.0.dylib \
            -change $BADPATH/libswscale.2.dylib @rpath/libswscale.2.dylib; done
        ;;
    windows-x86)
        # http://ffmpeg.org/platform.html#Linking-to-FFmpeg-with-Microsoft-Visual-C_002b_002b
        LIBS=(avcodec-55 avdevice-55 avfilter-4 avformat-55 avutil-52 postproc-52 swresample-0 swscale-2)
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
        LIBS=(avcodec-55 avdevice-55 avfilter-4 avformat-55 avutil-52 postproc-52 swresample-0 swscale-2)
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
