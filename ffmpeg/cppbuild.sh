#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" ffmpeg
    popd
    exit
fi

DISABLE="--disable-iconv --disable-opencl --disable-sdl2 --disable-bzlib --disable-lzma --disable-linux-perf --disable-xlib"
ENABLE="--enable-shared --enable-version3 --enable-runtime-cpudetect --enable-zlib --enable-libmp3lame --enable-libspeex --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libvo-amrwbenc --enable-openssl --enable-libopenh264 --enable-libvpx --enable-libfreetype --enable-libharfbuzz --enable-libopus --enable-libxml2 --enable-libsrt --enable-libwebp --enable-libaom --enable-libsvtav1 --enable-libzimg"
ENABLE_VULKAN="--enable-vulkan --enable-hwaccel=h264_vulkan --enable-hwaccel=hevc_vulkan --enable-hwaccel=av1_vulkan"

if [[ "$EXTENSION" == *gpl ]]; then
    # Enable GPLv3 modules
    ENABLE="$ENABLE --enable-gpl --enable-version3 --enable-libx264 --enable-libx265"
fi

# minimal configuration to support MPEG-4 streams with H.264 and AAC as well as Motion JPEG
# DISABLE="--disable-iconv --disable-libxcb --disable-opencl --disable-sdl2 --disable-bzlib --disable-lzma --disable-linux-perf --disable-everything"
# ENABLE="--enable-shared --enable-runtime-cpudetect --enable-libopenh264 --enable-encoder=libopenh264 --enable-encoder=aac --enable-encoder=mjpeg --enable-decoder=h264 --enable-decoder=aac --enable-decoder=mjpeg --enable-parser=h264 --enable-parser=aac --enable-parser=mjpeg --enable-muxer=mp4 --enable-muxer=rtsp --enable-muxer=mjpeg --enable-demuxer=mov --enable-demuxer=rtsp --enable-demuxer=mjpeg --enable-protocol=file --enable-protocol=http --enable-protocol=rtp --enable-protocol=rtmp"

HARFBUZZ_CONFIG="--default-library=static -Dcairo=disabled -Dchafa=disabled -Dcoretext=disabled -Ddirectwrite=disabled -Ddocs=disabled -Dfreetype=enabled -Dglib=disabled -Dgobject=disabled -Dgraphite=disabled -Dicu=disabled -Dtests=disabled -Dintrospection=disabled --libdir=lib --buildtype=release"
LIBXML_CONFIG="--enable-static --disable-shared --without-iconv --without-python --without-lzma --with-pic"
SRT_CONFIG="-DENABLE_APPS:BOOL=OFF -DENABLE_ENCRYPTION:BOOL=ON -DENABLE_SHARED:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_INCLUDEDIR=include -DCMAKE_INSTALL_BINDIR=bin"
WEBP_CONFIG="-DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_EXTRAS=OFF -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_WEBP_JS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib"
LIBAOM_CONFIG="-DENABLE_TESTS:BOOL=OFF -DENABLE_TESTDATA:BOOL=OFF -DENABLE_TOOLS:BOOL=OFF -DENABLE_EXAMPLES:BOOL=OFF -DENABLE_DOCS:BOOL=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_INCLUDEDIR=include -DCMAKE_INSTALL_BINDIR=bin"
LIBSVTAV1_CONFIG="-DBUILD_APPS:BOOL=OFF -DBUILD_TESTING:BOOL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_INCLUDEDIR=include -DCMAKE_INSTALL_BINDIR=bin"

NASM_VERSION=2.14
ZLIB=zlib-1.3.1
LAME=lame-3.100
SPEEX=speex-1.2.1
OPUS=opus-1.3.1
OPENCORE_AMR=opencore-amr-0.1.6
VO_AMRWBENC=vo-amrwbenc-0.1.3
OPENSSL=openssl-3.5.4
OPENH264_VERSION=2.6.0
X264=x264-stable
X265=3.4
VPX_VERSION=1.15.2
ALSA_VERSION=1.2.14
FREETYPE_VERSION=2.14.1
HARFBUZZ_VERSION=12.3.0
MFX_VERSION=1.35.1
NVCODEC_VERSION=13.0.19.0
XML2=libxml2-2.9.12
LIBSRT_VERSION=1.5.4
WEBP_VERSION=1.6.0
AOMAV1_VERSION=3.9.1
SVTAV1_VERSION=3.1.2
ZIMG_VERSION=3.0.6
FFMPEG_VERSION=8.0.1
download https://download.videolan.org/contrib/nasm/nasm-$NASM_VERSION.tar.gz nasm-$NASM_VERSION.tar.gz
download https://zlib.net/$ZLIB.tar.gz $ZLIB.tar.gz
download https://downloads.sourceforge.net/project/lame/lame/3.100/$LAME.tar.gz $LAME.tar.gz
download https://ftp.osuosl.org/pub/xiph/releases/speex/$SPEEX.tar.gz $SPEEX.tar.gz
download https://archive.mozilla.org/pub/opus/$OPUS.tar.gz $OPUS.tar.gz
download https://sourceforge.net/projects/opencore-amr/files/opencore-amr/$OPENCORE_AMR.tar.gz/download $OPENCORE_AMR.tar.gz
download https://sourceforge.net/projects/opencore-amr/files/vo-amrwbenc/$VO_AMRWBENC.tar.gz/download $VO_AMRWBENC.tar.gz
download https://www.openssl.org/source/$OPENSSL.tar.gz $OPENSSL.tar.gz
download https://github.com/cisco/openh264/archive/v$OPENH264_VERSION.tar.gz openh264-$OPENH264_VERSION.tar.gz
download https://code.videolan.org/videolan/x264/-/archive/stable/$X264.tar.gz $X264.tar.gz
download https://github.com/videolan/x265/archive/$X265.tar.gz x265-$X265.tar.gz
download https://github.com/webmproject/libvpx/archive/v$VPX_VERSION.tar.gz libvpx-$VPX_VERSION.tar.gz
download https://ftp.osuosl.org/pub/blfs/conglomeration/alsa-lib/alsa-lib-$ALSA_VERSION.tar.bz2 alsa-lib-$ALSA_VERSION.tar.bz2
download https://ftp.osuosl.org/pub/blfs/conglomeration/freetype/freetype-$FREETYPE_VERSION.tar.xz freetype-$FREETYPE_VERSION.tar.xz
download https://github.com/harfbuzz/harfbuzz/archive/refs/tags/$HARFBUZZ_VERSION.tar.gz harfbuzz-$HARFBUZZ_VERSION.tar.gz
download https://github.com/lu-zero/mfx_dispatch/archive/$MFX_VERSION.tar.gz mfx_dispatch-$MFX_VERSION.tar.gz
download http://xmlsoft.org/sources/$XML2.tar.gz $XML2.tar.gz
download https://github.com/Haivision/srt/archive/refs/tags/v$LIBSRT_VERSION.tar.gz srt-$LIBSRT_VERSION.tar.gz
download https://github.com/FFmpeg/nv-codec-headers/archive/n$NVCODEC_VERSION.tar.gz nv-codec-headers-$NVCODEC_VERSION.tar.gz
download https://github.com/webmproject/libwebp/archive/refs/tags/v$WEBP_VERSION.tar.gz libwebp-$WEBP_VERSION.tar.gz
download https://storage.googleapis.com/aom-releases/libaom-$AOMAV1_VERSION.tar.gz aom-$AOMAV1_VERSION.tar.gz
download https://gitlab.com/AOMediaCodec/SVT-AV1/-/archive/v$SVTAV1_VERSION/SVT-AV1-v$SVTAV1_VERSION.tar.gz SVT-AV1-$SVTAV1_VERSION.tar.gz
download https://github.com/sekrit-twc/zimg/archive/refs/tags/release-$ZIMG_VERSION.tar.gz zimg-release-$ZIMG_VERSION.tar.gz
download https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 ffmpeg-$FFMPEG_VERSION.tar.bz2

mkdir -p $PLATFORM$EXTENSION
cd $PLATFORM$EXTENSION
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
tar --totals -xzf ../srt-$LIBSRT_VERSION.tar.gz
tar --totals -xzf ../openh264-$OPENH264_VERSION.tar.gz
tar --totals -xzf ../$X264.tar.gz
tar --totals -xzf ../x265-$X265.tar.gz
tar --totals -xzf ../libvpx-$VPX_VERSION.tar.gz
tar --totals -xJf ../freetype-$FREETYPE_VERSION.tar.xz
tar --totals -xzf ../harfbuzz-$HARFBUZZ_VERSION.tar.gz
tar --totals -xzf ../mfx_dispatch-$MFX_VERSION.tar.gz
tar --totals -xzf ../nv-codec-headers-$NVCODEC_VERSION.tar.gz
tar --totals -xzf ../$XML2.tar.gz
tar --totals -xzf ../libwebp-$WEBP_VERSION.tar.gz
tar --totals -xzf ../aom-$AOMAV1_VERSION.tar.gz
tar --totals -xzf ../SVT-AV1-$SVTAV1_VERSION.tar.gz
tar --totals -xzf ../zimg-release-$ZIMG_VERSION.tar.gz
tar --totals -xjf ../ffmpeg-$FFMPEG_VERSION.tar.bz2

if [[ "${ACLOCAL_PATH:-}" == C:\\msys64\\* ]]; then
    export ACLOCAL_PATH=/mingw64/share/aclocal:/usr/share/aclocal
fi

cd nasm-$NASM_VERSION
# fix for build with GCC 8.x
sedinplace '/^\s*#\s*typedef _Bool bool/d' include/compiler.h
sedinplace 's/void pure_func/void/g' include/nasmlib.h
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ V=0
make install
cd ..

export PATH=$INSTALL_PATH/bin:$PATH
export PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/

patch -Np1 -d $LAME < ../../lame.patch
# patch -Np1 -d $OPENSSL < ../../openssl-android.patch
patch -Np1 -d ffmpeg-$FFMPEG_VERSION < ../../ffmpeg.patch
patch -Np1 -d ffmpeg-$FFMPEG_VERSION < ../../ffmpeg-vulkan.patch
# patch -Np1 -d ffmpeg-$FFMPEG_VERSION < ../../ffmpeg-flv-support-hevc-opus.patch
sedinplace 's/bool bEnableavx512/bool bEnableavx512 = false/g' x265-*/source/common/param.h
sedinplace 's/detect512()/false/g' x265-*/source/common/quant.cpp
sedinplace 's/CMAKE_C_COMPILER_ID MATCHES "Clang"/FALSE/g' SVT-AV1-*/CMakeLists.txt
# sedinplace 's/defined(__linux__)/defined(__linux__) \&\& !defined(__ANDROID__)/g' SVT-AV1-*/Source/Lib/Common/Codec/EbThreads.h
sedinplace '/ANativeWindow_release/d' ffmpeg-*/libavutil/hwcontext_mediacodec.c
sedinplace 's/#define MAX_SLICES 32/#define MAX_SLICES 256/g' ffmpeg-*/libavcodec/h264dec.h

case $PLATFORM in
    android-arm)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export CXX="$ANDROID_CC++ $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic  --host=arm-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=arm-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=arm-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --host=arm-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux
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
        PATH="${ANDROID_CC%/*}:$ANDROID_BIN/bin:$PATH" ./Configure --prefix=$INSTALL_PATH --libdir=lib android-arm no-shared no-tests -D__ANDROID_API__=24
        PATH="${ANDROID_CC%/*}:$ANDROID_BIN/bin:$PATH" ANDROID_DEV="$ANDROID_ROOT/usr" make -s -j $MAKEJ
        make install_dev
        cd ../srt-$LIBSRT_VERSION
        patch -Np1 < ../../../srt-android.patch || true
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../openh264-$OPENH264_VERSION
        sedinplace 's/stlport_shared/system/g' codec/build/android/dec/jni/Application.mk build/platform-android.mk
        sedinplace 's/12/24/g' codec/build/android/dec/jni/Application.mk build/platform-android.mk
        CFLAGS="$ANDROID_FLAGS" LDFLAGS="$ANDROID_FLAGS" make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=arm USE_ASM=No NDKROOT="$ANDROID_NDK" NDK_TOOLCHAIN_VERSION="clang" TARGET="android-24" install-static
        cd ../$X264
        patch -Np1 < ../../../x264-android.patch || true
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_PREFIX-" --sysroot="$ANDROID_ROOT" --host=arm-linux
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        patch -Np1 < ../../../x265-android.patch || true
        cd build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release
        make -j $MAKEJ

        cd ../10bit
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release
        make -j $MAKEJ

        cd ../8bit
        ln -sf ../10bit/libx265.a libx265_main10.a
        ln -sf ../12bit/libx265.a libx265_main12.a
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
ar -M <<EOF
CREATE libx265.a
ADDLIB libx265_main.a
ADDLIB libx265_main10.a
ADDLIB libx265_main12.a
SAVE
END
EOF
        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        patch -Np1 < ../../../libvpx-android.patch
        CFLAGS="$ANDROID_FLAGS" CXXFLAGS="$ANDROID_FLAGS" LDFLAGS="$ANDROID_FLAGS" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --disable-tools --target=armv7-android-gcc --disable-runtime-cpu-detect --disable-neon --disable-neon-asm
        make -j $MAKEJ || true
        sedinplace 's/_neon/_c/g' vpx_dsp_rtcd.h vpx_scale_rtcd.h vp8_rtcd.h vp9_rtcd.h
        sedinplace 's/vp8_loop_filter_mbhs_c/vp8_loop_filter_simple_horizontal_edge_c/g' vp8_rtcd.h
        sedinplace 's/vp8_loop_filter_mbvs_c/vp8_loop_filter_simple_vertical_edge_c/g' vp8_rtcd.h
        make -j $MAKEJ
        make install
        cd ../libwebp-$WEBP_VERSION
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=arm-linux
        make -j $MAKEJ
        make install
        cd ../harfbuzz-$HARFBUZZ_VERSION
        echo "[binaries]" >> android-arm.ini
        echo "c = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-android24-clang'" >> android-arm.ini
        echo "cpp = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-android24-clang++'" >> android-arm.ini
        echo "ar = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar'" >> android-arm.ini
        echo "windres = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-windres'" >> android-arm.ini
        echo "strip = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip'" >> android-arm.ini
        echo "pkg-config = '/usr/bin/pkg-config'" >> android-arm.ini
        echo "" >> android-arm.ini
        echo "[host_machine]" >> android-arm.ini
        echo "system = 'android'" >> android-arm.ini
        echo "cpu_family = 'arm'" >> android-arm.ini
        echo "cpu = 'armv7'" >> android-arm.ini
        echo "endian = 'little'" >> android-arm.ini
        meson setup build --prefix=$INSTALL_PATH --cross-file=android-arm.ini $HARFBUZZ_CONFIG --pkg-config-path=/usr/bin/pkg-config
        cd build
        meson compile
        meson install
        cd ..
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBAOM_CONFIG -DCONFIG_RUNTIME_CPU_DETECT:BOOL=OFF ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        sedinplace 's/unsigned long int/unsigned int/g' libavdevice/v4l2.c
        LDEXEFLAGS='-Wl,-rpath,\$$ORIGIN/' ./configure --prefix=.. $DISABLE $ENABLE --enable-jni --enable-mediacodec --enable-pthreads --enable-cross-compile --cross-prefix="$ANDROID_PREFIX-" --ar="$AR" --ranlib="$RANLIB" --cc="$CC" --strip="$STRIP" --sysroot="$ANDROID_ROOT" --target-os=android --arch=arm --extra-cflags="-I../include/ -I../include/libxml2 -I../include/mfx -I../include/svt-av1 $ANDROID_FLAGS" --extra-ldflags="-L../lib/ $ANDROID_FLAGS" --extra-libs="$ANDROID_LIBS -lz -latomic" --disable-symver  --pkg-config=/usr/bin/pkg-config || cat ffbuild/config.log
        make -j $MAKEJ
        make install
        ;;

    android-arm64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export CXX="$ANDROID_CC++ $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic  --host=aarch64-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=aarch64-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=aarch64-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --host=aarch64-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux
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
        PATH="${ANDROID_CC%/*}:$ANDROID_BIN/bin:$PATH" ./Configure --prefix=$INSTALL_PATH --libdir=lib android-arm64 no-shared no-tests -D__ANDROID_API__=24
        PATH="${ANDROID_CC%/*}:$ANDROID_BIN/bin:$PATH" ANDROID_DEV="$ANDROID_ROOT/usr" make -s -j $MAKEJ
        make install_dev
        cd ../srt-$LIBSRT_VERSION
        patch -Np1 < ../../../srt-android.patch || true
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../openh264-$OPENH264_VERSION
        sedinplace 's/stlport_shared/system/g' codec/build/android/dec/jni/Application.mk build/platform-android.mk
        sedinplace 's/12/24/g' codec/build/android/dec/jni/Application.mk build/platform-android.mk
        CFLAGS="$ANDROID_FLAGS" LDFLAGS="$ANDROID_FLAGS" make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=arm64 USE_ASM=No NDKROOT="$ANDROID_NDK" NDK_TOOLCHAIN_VERSION="clang" TARGET="android-24" install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_PREFIX-" --sysroot="$ANDROID_ROOT" --host=aarch64-linux
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        patch -Np1 < ../../../x265-android.patch || true
        cd build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release
        make -j $MAKEJ

        cd ../10bit
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release
        make -j $MAKEJ

        cd ../8bit
        ln -sf ../10bit/libx265.a libx265_main10.a
        ln -sf ../12bit/libx265.a libx265_main12.a
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
ar -M <<EOF
CREATE libx265.a
ADDLIB libx265_main.a
ADDLIB libx265_main10.a
ADDLIB libx265_main12.a
SAVE
END
EOF
        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        patch -Np1 < ../../../libvpx-android.patch
        CFLAGS="$ANDROID_FLAGS" CXXFLAGS="$ANDROID_FLAGS" LDFLAGS="$ANDROID_FLAGS" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --disable-tools --target=arm64-android-gcc --disable-runtime-cpu-detect --disable-neon --disable-neon-asm
        make -j $MAKEJ || true
        sedinplace 's/_neon/_c/g' vpx_dsp_rtcd.h vpx_scale_rtcd.h vp8_rtcd.h vp9_rtcd.h
        sedinplace 's/vp8_loop_filter_mbhs_c/vp8_loop_filter_simple_horizontal_edge_c/g' vp8_rtcd.h
        sedinplace 's/vp8_loop_filter_mbvs_c/vp8_loop_filter_simple_vertical_edge_c/g' vp8_rtcd.h
        make -j $MAKEJ
        make install
        cd ../libwebp-$WEBP_VERSION
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=aarch64-linux
        make -j $MAKEJ
        make install
        cd ../harfbuzz-$HARFBUZZ_VERSION
        echo "[binaries]" >> android-arm.ini
        echo "c = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang'" >> android-arm.ini
        echo "cpp = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++'" >> android-arm.ini
        echo "ar = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar'" >> android-arm.ini
        echo "windres = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-windres'" >> android-arm.ini
        echo "strip = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip'" >> android-arm.ini
        echo "pkg-config = '/usr/bin/pkg-config'" >> android-arm.ini
        echo "[host_machine]" >> android-arm.ini
        echo "system = 'android'" >> android-arm.ini
        echo "cpu_family = 'aarch64'" >> android-arm.ini
        echo "cpu = 'aarch64'" >> android-arm.ini
        echo "endian = 'little'" >> android-arm.ini
        meson setup build --prefix=$INSTALL_PATH --cross-file=android-arm.ini $HARFBUZZ_CONFIG --pkg-config-path=/usr/bin/pkg-config
        cd build
        meson compile
        meson install
        cd ..
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBAOM_CONFIG -DCONFIG_RUNTIME_CPU_DETECT:BOOL=OFF -DENABLE_NEON_I8MM=OFF ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        sedinplace 's/unsigned long int/unsigned int/g' libavdevice/v4l2.c
        LDEXEFLAGS='-Wl,-rpath,\$$ORIGIN/' ./configure --prefix=.. $DISABLE $ENABLE --enable-jni --enable-mediacodec --enable-pthreads --enable-cross-compile --cross-prefix="$ANDROID_PREFIX-" --ar="$AR" --ranlib="$RANLIB" --cc="$CC" --strip="$STRIP" --sysroot="$ANDROID_ROOT" --target-os=android --arch=aarch64 --extra-cflags="-I../include/ -I../include/libxml2 -I../include/mfx -I../include/svt-av1 $ANDROID_FLAGS" --extra-ldflags="-L../lib/ $ANDROID_FLAGS" --extra-libs="$ANDROID_LIBS -lz -latomic" --disable-symver --pkg-config=/usr/bin/pkg-config || cat ffbuild/config.log
        make -j $MAKEJ
        make install
        ;;

     android-x86)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export CXX="$ANDROID_CC++ $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=i686-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --host=i686-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
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
        PATH="${ANDROID_CC%/*}:$ANDROID_BIN/bin:$PATH" ./Configure --prefix=$INSTALL_PATH --libdir=lib android-x86 no-shared no-tests -D__ANDROID_API__=24
        PATH="${ANDROID_CC%/*}:$ANDROID_BIN/bin:$PATH" ANDROID_DEV="$ANDROID_ROOT/usr" make -s -j $MAKEJ
        make install_dev
        cd ../srt-$LIBSRT_VERSION
        patch -Np1 < ../../../srt-android.patch || true
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../openh264-$OPENH264_VERSION
        sedinplace 's/stlport_shared/system/g' codec/build/android/dec/jni/Application.mk build/platform-android.mk
        sedinplace 's/12/24/g' codec/build/android/dec/jni/Application.mk build/platform-android.mk
        CFLAGS="$ANDROID_FLAGS" LDFLAGS="$ANDROID_FLAGS" make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=x86 USE_ASM=No NDKROOT="$ANDROID_NDK" NDK_TOOLCHAIN_VERSION="clang" TARGET="android-24" install-static
        cd ../$X264
        patch -Np1 < ../../../x264-android.patch || true
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_PREFIX-" --sysroot="$ANDROID_ROOT" --host=i686-linux --disable-asm
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        patch -Np1 < ../../../x265-android.patch || true
        cd build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm
        make -j $MAKEJ

        cd ../10bit
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm
        make -j $MAKEJ

        cd ../8bit
        ln -sf ../10bit/libx265.a libx265_main10.a
        ln -sf ../12bit/libx265.a libx265_main12.a
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
ar -M <<EOF
CREATE libx265.a
ADDLIB libx265_main.a
ADDLIB libx265_main10.a
ADDLIB libx265_main12.a
SAVE
END
EOF
        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        patch -Np1 < ../../../libvpx-android.patch
        CFLAGS="$ANDROID_FLAGS" CXXFLAGS="$ANDROID_FLAGS" LDFLAGS="$ANDROID_FLAGS" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --disable-tools --target=x86-android-gcc --as=nasm
        make -j $MAKEJ
        make install
        cd ../libwebp-$WEBP_VERSION
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../harfbuzz-$HARFBUZZ_VERSION
        echo "[binaries]" >> android-i386.ini
        echo "c = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android24-clang'" >> android-i386.ini
        echo "cpp = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android24-clang++'" >> android-i386.ini
        echo "ar = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar'" >> android-i386.ini
        echo "windres = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-windres'" >> android-i386.ini
        echo "strip = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip'" >> android-i386.ini
        echo "pkg-config = '/usr/bin/pkg-config'" >> android-i386.ini
        echo "[host_machine]" >> android-i386.ini
        echo "system = 'android'" >> android-i386.ini
        echo "cpu_family = 'x86'" >> android-i386.ini
        echo "cpu = 'x86'" >> android-i386.ini
        echo "endian = 'little'" >> android-i386.ini
        meson setup build --prefix=$INSTALL_PATH --cross-file=android-i386.ini $HARFBUZZ_CONFIG --pkg-config-path=/usr/bin/pkg-config
        cd build
        meson compile
        meson install
        cd ..
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBAOM_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        sedinplace 's/unsigned long int/unsigned int/g' libavdevice/v4l2.c
        LDEXEFLAGS='-Wl,-rpath,\$$ORIGIN/' ./configure --prefix=.. $DISABLE $ENABLE --enable-jni --enable-mediacodec --enable-pthreads --enable-cross-compile --cross-prefix="$ANDROID_PREFIX-" --ar="$AR" --ranlib="$RANLIB" --cc="$CC" --strip="$STRIP" --sysroot="$ANDROID_ROOT" --target-os=android --arch=atom --extra-cflags="-I../include/ -I../include/libxml2 -I../include/mfx -I../include/svt-av1 $ANDROID_FLAGS" --extra-ldflags="-L../lib/ $ANDROID_FLAGS" --extra-libs="$ANDROID_LIBS -lz -latomic" --disable-symver --pkg-config=/usr/bin/pkg-config || cat ffbuild/config.log
        make -j $MAKEJ
        make install
        ;;

     android-x86_64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export CXX="$ANDROID_CC++ $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic  --host=x86_64-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        ./configure --prefix=$INSTALL_PATH --static --uname=x86_64-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --host=x86_64-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
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
        PATH="${ANDROID_CC%/*}:$ANDROID_BIN/bin:$PATH" ./Configure --prefix=$INSTALL_PATH --libdir=lib android-x86_64 no-shared no-tests -D__ANDROID_API__=24
        PATH="${ANDROID_CC%/*}:$ANDROID_BIN/bin:$PATH" ANDROID_DEV="$ANDROID_ROOT/usr" make -s -j $MAKEJ
        make install_dev
        cd ../srt-$LIBSRT_VERSION
        patch -Np1 < ../../../srt-android.patch || true
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../openh264-$OPENH264_VERSION
        sedinplace 's/stlport_shared/system/g' codec/build/android/dec/jni/Application.mk build/platform-android.mk
        sedinplace 's/12/24/g' codec/build/android/dec/jni/Application.mk build/platform-android.mk
        CFLAGS="$ANDROID_FLAGS" LDFLAGS="$ANDROID_FLAGS" make -j $MAKEJ PREFIX=$INSTALL_PATH OS=android ARCH=x86_64 USE_ASM=No NDKROOT="$ANDROID_NDK" NDK_TOOLCHAIN_VERSION="clang" TARGET="android-24" install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_PREFIX-" --sysroot="$ANDROID_ROOT" --host=x86_64-linux
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265
        patch -Np1 < ../../../x265-android.patch || true
        cd build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm
        make -j $MAKEJ

        cd ../10bit
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm
        make -j $MAKEJ

        cd ../8bit
        ln -sf ../10bit/libx265.a libx265_main10.a
        ln -sf ../12bit/libx265.a libx265_main12.a
        $CMAKE ../../../source -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm -DENABLE_CLI=OFF
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
ar -M <<EOF
CREATE libx265.a
ADDLIB libx265_main.a
ADDLIB libx265_main10.a
ADDLIB libx265_main12.a
SAVE
END
EOF
        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        patch -Np1 < ../../../libvpx-android.patch
        CFLAGS="$ANDROID_FLAGS" CXXFLAGS="$ANDROID_FLAGS" LDFLAGS="$ANDROID_FLAGS" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --disable-tools --target=x86_64-android-gcc --as=nasm
        make -j $MAKEJ
        make install
        cd ../libwebp-$WEBP_VERSION
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ
        make install
        cd ../harfbuzz-$HARFBUZZ_VERSION
        echo "[binaries]" >> android-amd64.ini
        echo "c = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android24-clang'" >> android-amd64.ini
        echo "cpp = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android24-clang++'" >> android-amd64.ini
        echo "ar = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar'" >> android-amd64.ini
        echo "windres = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-windres'" >> android-amd64.ini
        echo "strip = '${PLATFORM_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip'" >> android-amd64.ini
        echo "pkg-config = '/usr/bin/pkg-config'" >> android-amd64.ini
        echo "[host_machine]" >> android-amd64.ini
        echo "system = 'android'" >> android-amd64.ini
        echo "cpu_family = 'x86_64'" >> android-amd64.ini
        echo "cpu = 'x86_64'" >> android-amd64.ini
        echo "endian = 'little'" >> android-amd64.ini
        meson setup build --prefix=$INSTALL_PATH --cross-file=android-amd64.ini $HARFBUZZ_CONFIG --pkg-config-path=/usr/bin/pkg-config
        cd build
        meson compile
        meson install
        cd ..
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBAOM_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/include/" -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/lib/" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        sedinplace 's/unsigned long int/unsigned int/g' libavdevice/v4l2.c
        LDEXEFLAGS='-Wl,-rpath,\$$ORIGIN/' ./configure --prefix=.. $DISABLE $ENABLE --enable-jni --enable-mediacodec --enable-pthreads --enable-cross-compile --cross-prefix="$ANDROID_PREFIX-" --ar="$AR" --ranlib="$RANLIB" --cc="$CC" --strip="$STRIP" --sysroot="$ANDROID_ROOT" --target-os=android --arch=atom --extra-cflags="-I../include/ -I../include/libxml2 -I../include/mfx -I../include/svt-av1 $ANDROID_FLAGS" --extra-ldflags="-L../lib/ $ANDROID_FLAGS" --extra-libs="$ANDROID_LIBS -lz -latomic" --disable-symver --pkg-config=/usr/bin/pkg-config || cat ffbuild/config.log
        make -j $MAKEJ
        make install
        ;;

    linux-x86)
        export AS="nasm"
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        CC="gcc -m32 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=i686-linux CFLAGS="-m32 -msse2"
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --host=i686-linux CFLAGS="-m32 -msse2"
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux CFLAGS="-m32"
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
        ./Configure linux-elf -m32 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../srt-$LIBSRT_VERSION
        CC="gcc -m32" CXX="g++ -m32" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86 USE_ASM=No install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=i686-linux
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265/build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        CC="gcc -m32" CXX="g++ -m32" $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm
        make -j $MAKEJ

        cd ../10bit
        CC="gcc -m32" CXX="g++ -m32" $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm
        make -j $MAKEJ

        cd ../8bit
        ln -sf ../10bit/libx265.a libx265_main10.a
        ln -sf ../12bit/libx265.a libx265_main12.a
        CC="gcc -m32" CXX="g++ -m32" $CMAKE ../../../source -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm -DENABLE_CLI=OFF
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
ar -M <<EOF
CREATE libx265.a
ADDLIB libx265_main.a
ADDLIB libx265_main10.a
ADDLIB libx265_main12.a
SAVE
END
EOF
        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=x86-linux-gcc --as=nasm
        make -j $MAKEJ
        make install
        cd ../libwebp-$WEBP_VERSION
        CC="gcc -m32 -fPIC" CXX="g++ -m32 -fPIC" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=i686-linux CFLAGS="-m32"
        make -j $MAKEJ
        make install
        cd ../harfbuzz-$HARFBUZZ_VERSION
        meson setup build --prefix=$INSTALL_PATH $HARFBUZZ_CONFIG -Dc_args="-m32" -Dcpp_args="-m32"
        cd build
        meson compile
        meson install
        cd ..
        LIBS=
        if [[ ! -z $(ldconfig -p | grep libva-drm) ]]; then
            cd ../mfx_dispatch-$MFX_VERSION
            autoreconf -fiv
            PKG_CONFIG_PATH="../lib/pkgconfig" ./configure --prefix=$INSTALL_PATH --disable-shared --enable-static --enable-fast-install --with-pic --host=i686-linux CFLAGS="-m32 -D__ILP32__" CXXFLAGS="-m32 -D__ILP32__ -std=c++11"
            make -j $MAKEJ
            make install
            ENABLE="$ENABLE --enable-libmfx"
            LIBS="-lva-drm -lva-x11 -lva"
        fi
        cd ../nv-codec-headers-n$NVCODEC_VERSION
        make install PREFIX=$INSTALL_PATH
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        CC="gcc -m32 -fPIC" CXX="g++ -m32 -fPIC" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBAOM_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        CC="gcc -m32 -fPIC" CXX="g++ -m32 -fPIC" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        LDEXEFLAGS='-Wl,-rpath,\$$ORIGIN/' PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $ENABLE_VULKAN --enable-libdrm --enable-cuda --enable-cuvid --enable-nvenc --enable-pthreads --enable-libxcb --enable-libpulse --cc="gcc -m32 -D__ILP32__" --extra-cflags="-I../include/ -I../include/libxml2 -I../include/mfx -I../include/svt-av1" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -lpthread -ldl -lz -lm $LIBS" || cat ffbuild/config.log
        make -j $MAKEJ
        make install
        ;;

    linux-x86_64)
        export AS="nasm"
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        CC="gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --host=x86_64-linux CFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
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
        ./Configure linux-x86_64 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../srt-$LIBSRT_VERSION
        CC="gcc -m64" CXX="g++ -m64" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86_64 USE_ASM=No install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=x86_64-linux
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265/build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm
        make -j $MAKEJ

        cd ../10bit
        $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm
        make -j $MAKEJ

        cd ../8bit
        ln -sf ../10bit/libx265.a libx265_main10.a
        ln -sf ../12bit/libx265.a libx265_main12.a
        $CMAKE ../../../source -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm -DENABLE_CLI=OFF
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
ar -M <<EOF
CREATE libx265.a
ADDLIB libx265_main.a
ADDLIB libx265_main10.a
ADDLIB libx265_main12.a
SAVE
END
EOF
        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=x86_64-linux-gcc --as=nasm
        make -j $MAKEJ
        make install
        cd ../libwebp-$WEBP_VERSION
        CC="gcc -m64 -fPIC" CXX="g++ -m64 -fPIC" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=x86_64-linux CFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../harfbuzz-$HARFBUZZ_VERSION
        meson setup build --prefix=$INSTALL_PATH $HARFBUZZ_CONFIG
        cd build
        meson compile
        meson install
        cd ..
        LIBS=
        if [[ ! -z $(ldconfig -p | grep libva-drm) ]]; then
            cd ../mfx_dispatch-$MFX_VERSION
            autoreconf -fiv
            PKG_CONFIG_PATH="../lib/pkgconfig" ./configure --prefix=$INSTALL_PATH --disable-shared --enable-static --enable-fast-install --with-pic --host=x86_64-linux CFLAGS="-m64" CXXFLAGS="-m64"
            make -j $MAKEJ
            make install
            ENABLE="$ENABLE --enable-libmfx"
            LIBS="-lva-drm -lva-x11 -lva"
        fi
        cd ../nv-codec-headers-n$NVCODEC_VERSION
        make install PREFIX=$INSTALL_PATH
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        CC="gcc -m64" CXX="g++ -m64" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBAOM_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        pwd
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        CC="gcc -m64" CXX="g++ -m64" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        LDEXEFLAGS='-Wl,-rpath,\$$ORIGIN/' PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $ENABLE_VULKAN --enable-libdrm --enable-cuda --enable-cuvid --enable-nvenc --enable-pthreads --enable-libxcb --enable-libpulse --cc="gcc -m64" --extra-cflags="-I../include/ -I../include/libxml2 -I../include/mfx -I../include/svt-av1" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -lpthread -ldl -lz -lm $LIBS" || cat ffbuild/config.log
        make -j $MAKEJ
        make install
        ;;

    linux-armhf)
        tar --totals -xjf ../alsa-lib-$ALSA_VERSION.tar.bz2

        export CFLAGS="-I$INSTALL_PATH/include -L$INSTALL_PATH/lib"
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

        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        CC="arm-linux-gnueabihf-gcc -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --host=arm-linux-gnueabihf
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf
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
          ./Configure linux-generic32 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib --cross-compile-prefix=arm-linux-gnueabihf-
        else
          ./Configure linux-generic32 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        fi
        make -s -j $MAKEJ
        make install_sw
        cd ../srt-$LIBSRT_VERSION
        if [ $CROSSCOMPILE -eq 1 ]
        then
          $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ -DCMAKE_STRIP=arm-linux-gnueabihf-strip -DCMAKE_FIND_ROOT_PATH=arm-linux-gnueabihf .
        else
          $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG .
        fi
        make -j $MAKEJ V=0
        make install
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
        cd ../x265-$X265/build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        if [ $CROSSCOMPILE -eq 1 ]
        then
          $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_STRIP=arm-linux-gnueabihf-strip -DCMAKE_FIND_ROOT_PATH=arm-linux-gnueabihf -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
          make -j $MAKEJ

          cd ../10bit
          $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_STRIP=arm-linux-gnueabihf-strip -DCMAKE_FIND_ROOT_PATH=arm-linux-gnueabihf -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
          make -j $MAKEJ

          cd ../8bit
          ln -sf ../10bit/libx265.a libx265_main10.a
          ln -sf ../12bit/libx265.a libx265_main12.a
          $CMAKE ../../../source -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_STRIP=arm-linux-gnueabihf-strip -DCMAKE_FIND_ROOT_PATH=arm-linux-gnueabihf -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF -DENABLE_CLI=OFF
        else
          $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
          make -j $MAKEJ

          cd ../10bit
          $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
          make -j $MAKEJ

          cd ../8bit
          ln -sf ../10bit/libx265.a libx265_main10.a
          ln -sf ../12bit/libx265.a libx265_main12.a
          $CMAKE ../../../source -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF -DENABLE_CLI=OFF
        fi
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
ar -M <<EOF
CREATE libx265.a
ADDLIB libx265_main.a
ADDLIB libx265_main10.a
ADDLIB libx265_main12.a
SAVE
END
EOF
        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        patch -Np1 < ../../../libvpx-linux-arm.patch
        if [ $CROSSCOMPILE -eq 1 ]
        then
          CROSS=arm-linux-gnueabihf- ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=armv7-linux-gcc
        else
          ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests
        fi
        make -j $MAKEJ
        make install
        cd ../libwebp-$WEBP_VERSION
        if [ $CROSSCOMPILE -eq 1 ]
        then
          $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS -fPIC" -DCMAKE_C_FLAGS="$CFLAGS -fPIC" -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ -DCMAKE_STRIP=arm-linux-gnueabihf-strip -DCMAKE_FIND_ROOT_PATH=arm-linux-gnueabih .
        else
          $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG .
        fi
        make -j $MAKEJ V=0
        make -j $MAKEJ V=0
        make install
        cd ../alsa-lib-$ALSA_VERSION/
        ./configure --host=arm-linux-gnueabihf --prefix=$INSTALL_PATH --disable-python
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install
        cd ../harfbuzz-$HARFBUZZ_VERSION
        if [ $CROSSCOMPILE -eq 1 ]
        then
            echo "[binaries]" >> linux-arm.ini
            echo "c = 'arm-linux-gnueabihf-gcc'" >> linux-arm.ini
            echo "cpp = 'arm-linux-gnueabihf-g++'" >> linux-arm.ini
            echo "ar = 'arm-linux-gnueabihf-ar'" >> linux-arm.ini
            echo "windres = 'arm-linux-gnueabihf-windres'" >> linux-arm.ini
            echo "strip = 'arm-linux-gnueabihf-strip'" >> linux-arm.ini
            echo "pkg-config='/usr/bin/pkg-config'" >> linux-arm.ini
            echo "" >> linux-arm.ini
            echo "[host_machine]" >> linux-arm.ini
            echo "system = 'linux'" >> linux-arm.ini
            echo "cpu_family = 'arm'" >> linux-arm.ini
            echo "cpu = 'armv6'" >> linux-arm.ini
            echo "endian = 'little'" >> linux-arm.ini
            PKG_CONFIG=/usr/bin/pkg-config
            PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig
            meson setup build --prefix=$INSTALL_PATH $HARFBUZZ_CONFIG --cross-file=linux-arm.ini --pkg-config-path=/usr/bin/pkg-config
        else
            PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig
            meson setup build --prefix=$INSTALL_PATH $HARFBUZZ_CONFIG
        fi
        cd build
        meson compile
        meson install
        cd ..
        cd ../nv-codec-headers-n$NVCODEC_VERSION
        make install PREFIX=$INSTALL_PATH
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        if [ $CROSSCOMPILE -eq 1 ]
        then
          $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS -fPIC" -DCMAKE_C_FLAGS="$CFLAGS -fPIC" -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ -DCMAKE_STRIP=arm-linux-gnueabihf-strip -DCMAKE_FIND_ROOT_PATH=arm-linux-gnueabih -DAOM_TARGET_CPU=generic $LIBAOM_CONFIG ..
        else
          $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBAOM_CONFIG ..
        fi
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        if [ $CROSSCOMPILE -eq 1 ]
        then
          $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 -DCMAKE_CXX_FLAGS="$CXXFLAGS -fPIC" -DCMAKE_C_FLAGS="$CFLAGS -fPIC" -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ -DCMAKE_STRIP=arm-linux-gnueabihf-strip -DCMAKE_FIND_ROOT_PATH=arm-linux-gnueabih $LIBSVTAV1_CONFIG ..
        else
          $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG ..
        fi
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        if [ $CROSSCOMPILE -eq 1 ]
        then
          if [[ ! -d $USERLAND_PATH ]]; then
            USERLAND_PATH="$(which arm-linux-gnueabihf-gcc | grep -o '.*/tools/')../userland"
          fi
          LDEXEFLAGS='-Wl,-rpath,\$$ORIGIN/' PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $ENABLE_VULKAN --enable-omx --enable-mmal --enable-omx-rpi --enable-pthreads --enable-libxcb --enable-libpulse --cc="arm-linux-gnueabihf-gcc" --extra-cflags="$CFLAGS -I$USERLAND_PATH/ -I$USERLAND_PATH/interface/vmcs_host/khronos/IL/ -I$USERLAND_PATH/host_applications/linux/libs/bcm_host/include/ -I../include/ -I../include/libxml2 -I../include/mfx/ -I../include/svt-av1" --extra-ldflags="-L$USERLAND_PATH/build/lib/ -L../lib/" --extra-libs="-lstdc++ -lasound -lvchiq_arm -lvcsm -lvcos -lpthread -ldl -lz -lm" --enable-cross-compile --arch=armhf --target-os=linux --cross-prefix="arm-linux-gnueabihf-" || cat ffbuild/config.log
        else
          LDEXEFLAGS='-Wl,-rpath,\$$ORIGIN/' PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $ENABLE_VULKAN --enable-cuda --enable-cuvid --enable-nvenc --enable-omx --enable-mmal --enable-omx-rpi --enable-pthreads --enable-libxcb --enable-libpulse --extra-cflags="-I../include/ -I../include/libxml2 -I../include/mfx/ -I../include/svt-av1" --extra-ldflags="-L../lib/ -L/opt/vc/lib" --extra-libs="-lstdc++ -lasound -lvchiq_arm -lvcsm -lvcos -lpthread -ldl -lz -lm" || cat ffbuild/config.log
        fi
        make -j $MAKEJ
        make install
        ;;

    linux-arm64)
        tar --totals -xjf ../alsa-lib-$ALSA_VERSION.tar.bz2

        export CFLAGS="-march=armv8-a+crypto -mcpu=cortex-a57+crypto -I$INSTALL_PATH/include -L$INSTALL_PATH/lib"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        HOST_ARCH="$(uname -m)"
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=aarch64-linux-gnu
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        CC="aarch64-linux-gnu-gcc -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=aarch64-linux-gnu
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --host=aarch64-linux-gnu
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux-gnu
        make -j $MAKEJ V=0
        make install
        cd ../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux-gnu
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux-gnu
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux-gnu
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        ./Configure linux-aarch64 -fPIC --prefix=$INSTALL_PATH --libdir=lib --cross-compile-prefix=aarch64-linux-gnu- "$CFLAGS" no-shared
        make -s -j $MAKEJ
        make install_sw
        cd ../srt-$LIBSRT_VERSION
        CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ .
        make -j $MAKEJ V=0
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. OS=linux ARCH=arm64 USE_ASM=No install-static CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++
        cd ../$X264
        LDFLAGS="-Wl,-z,relro" ./configure --prefix=$INSTALL_PATH --enable-pic --enable-static --disable-shared --disable-opencl --disable-cli --enable-asm --host=aarch64-linux-gnu --extra-cflags="$CFLAGS -fno-aggressive-loop-optimizations" --cross-prefix="aarch64-linux-gnu-"
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265/build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
        make -j $MAKEJ

        cd ../10bit
        $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
        make -j $MAKEJ

        cd ../8bit
        ln -sf ../10bit/libx265.a libx265_main10.a
        ln -sf ../12bit/libx265.a libx265_main12.a
        $CMAKE ../../../source -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF -DENABLE_CLI=OFF
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
ar -M <<EOF
CREATE libx265.a
ADDLIB libx265_main.a
ADDLIB libx265_main10.a
ADDLIB libx265_main12.a
SAVE
END
EOF
        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        patch -Np1 < ../../../libvpx-linux-arm.patch
        sedinplace '/neon_i8mm/d' ./configure
        CROSS=aarch64-linux-gnu- ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=armv8-linux-gcc
        sedinplace 's/HAS_NEON_I8MM (1 << 2)/HAS_NEON_I8MM 0/g' vpx_ports/*.h
        sedinplace 's/#if HAVE_NEON_I8MM/#if 0/g' test/*.c* vpx_ports/*.c*
        sedinplace 's/flags & HAS_NEON_I8MM/0/g' *.h
        make -j $MAKEJ
        make install
        cd ../libwebp-$WEBP_VERSION
        CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS -fPIC" -DCMAKE_C_FLAGS="$CFLAGS -fPIC" -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ .
        make -j $MAKEJ V=0
        make install
        cd ../alsa-lib-$ALSA_VERSION/
        ./configure --host=aarch64-linux-gnu --prefix=$INSTALL_PATH --disable-python
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=aarch64-linux-gnu
        make -j $MAKEJ
        make install
        cd ../harfbuzz-$HARFBUZZ_VERSION
        echo "[binaries]" >> linux-arm.ini
        echo "c = 'aarch64-linux-gnu-gcc'" >> linux-arm.ini
        echo "cpp = 'aarch64-linux-gnu-g++'" >> linux-arm.ini
        echo "ar = 'aarch64-linux-gnu-ar'" >> linux-arm.ini
        echo "windres = 'aarch64-linux-gnu-windres'" >> linux-arm.ini
        echo "strip = 'aarch64-linux-gnu-strip'" >> linux-arm.ini
        echo "pkg-config='/usr/bin/pkg-config'" >> linux-arm.ini
        echo "[properties]" >> linux-arm.ini
        echo "needs_exe_wrapper = true" >> linux-arm.ini
        echo "[host_machine]" >> linux-arm.ini
        echo "system = 'linux'" >> linux-arm.ini
        echo "cpu_family = 'aarch64'" >> linux-arm.ini
        echo "cpu = 'aarch64'" >> linux-arm.ini
        echo "endian = 'little'" >> linux-arm.ini
        meson setup build --prefix=$INSTALL_PATH $HARFBUZZ_CONFIG --cross-file=linux-arm.ini --pkg-config-path=/usr/bin/pkg-config
        cd build
        meson compile
        meson install
        cd ..
        cd ../nv-codec-headers-n$NVCODEC_VERSION
        make install PREFIX=$INSTALL_PATH
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS -fPIC" -DCMAKE_C_FLAGS="$CFLAGS -fPIC" -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ $LIBAOM_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS -fPIC" -DCMAKE_C_FLAGS="$CFLAGS -fPIC" -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ $LIBSVTAV1_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        if [[ ! -d $USERLAND_PATH ]]; then
          USERLAND_PATH="$(which aarch64-linux-gnu-gcc | grep -o '.*/tools/')../userland"
        fi
        LDEXEFLAGS='-Wl,-rpath,\$$ORIGIN/' PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $ENABLE_VULKAN --enable-cuda --enable-cuvid --enable-nvenc --enable-omx `#--enable-mmal` --enable-omx-rpi --enable-pthreads --enable-libxcb --enable-libpulse --cc="aarch64-linux-gnu-gcc" --extra-cflags="$CFLAGS -I$USERLAND_PATH/ -I$USERLAND_PATH/interface/vmcs_host/khronos/IL/ -I$USERLAND_PATH/host_applications/linux/libs/bcm_host/include/ -I../include/ -I../include/libxml2 -I../include/mfx/ -I../include/svt-av1 -fno-aggressive-loop-optimizations" --extra-ldflags="-Wl,-z,relro -L$USERLAND_PATH/build/lib/ -L../lib/" --extra-libs="-lstdc++ -lasound -lvchiq_arm `#-lvcsm` -lvcos -lpthread -ldl -lz -lm" --enable-cross-compile --arch=arm64 --target-os=linux --cross-prefix="aarch64-linux-gnu-" || cat ffbuild/config.log
        make -j $MAKEJ
        make install
        ;;

    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        else
          CC="powerpc64le-linux-gnu-gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=powerpc64le-linux-gnu
        fi
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        else
          CC="powerpc64le-linux-gnu-gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        fi
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64"
        else
          CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" ./configure --host=powerpc64le-linux-gnu --prefix=$INSTALL_PATH --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64"
        fi
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --build=ppc64le-linux CFLAGS="-m64"
        else
          CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" ./configure --host=powerpc64le-linux-gnu --prefix=$INSTALL_PATH $LIBXML_CONFIG --build=ppc64le-linux CFLAGS="-m64"
        fi
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=ppc64le-linux CFLAGS="-m64"
        else
          PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=powerpc64le-linux-gnu --build=ppc64le-linux CFLAGS="-m64"
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
          ./Configure linux-ppc64le -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        else
          ./Configure linux-ppc64le -fPIC no-shared --cross-compile-prefix=powerpc64le-linux-gnu- --prefix=$INSTALL_PATH --libdir=lib
        fi
        make -s -j $MAKEJ
        make install_sw
        cd ../srt-$LIBSRT_VERSION
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG .
        else
          CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=ppc64le -DCMAKE_CXX_FLAGS="-m64" -DCMAKE_C_FLAGS="-m64" -DCMAKE_C_COMPILER=powerpc64le-linux-gnu-gcc -DCMAKE_CXX_COMPILER=powerpc64le-linux-gnu-g++ -DCMAKE_STRIP=powerpc64le-linux-gnu-strip -DCMAKE_FIND_ROOT_PATH=powerpc64le-linux-gnu .
        fi
        make -j $MAKEJ V=0
        make install
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

        cd ../x265-$X265/build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64" CXX="g++ -m64" $CMAKE ../../../source -DENABLE_ALTIVEC=OFF -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
          make -j $MAKEJ

          cd ../10bit
          CC="gcc -m64" CXX="g++ -m64" $CMAKE ../../../source -DENABLE_ALTIVEC=OFF -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
          make -j $MAKEJ

          cd ../8bit
          ln -sf ../10bit/libx265.a libx265_main10.a
          ln -sf ../12bit/libx265.a libx265_main12.a
          CC="gcc -m64" CXX="g++ -m64" $CMAKE ../../../source -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF -DENABLE_CLI=OFF
        else
          $CMAKE ../../../source -DENABLE_ALTIVEC=OFF -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=ppc64le -DCMAKE_CXX_FLAGS="-m64" -DCMAKE_C_FLAGS="-m64" -DCMAKE_C_COMPILER=powerpc64le-linux-gnu-gcc -DCMAKE_CXX_COMPILER=powerpc64le-linux-gnu-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_STRIP=powerpc64le-linux-gnu-strip -DCMAKE_FIND_ROOT_PATH=powerpc64le-linux-gnu -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
          make -j $MAKEJ

          cd ../10bit
          $CMAKE ../../../source -DENABLE_ALTIVEC=OFF -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=ppc64le -DCMAKE_CXX_FLAGS="-m64" -DCMAKE_C_FLAGS="-m64" -DCMAKE_C_COMPILER=powerpc64le-linux-gnu-gcc -DCMAKE_CXX_COMPILER=powerpc64le-linux-gnu-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_STRIP=powerpc64le-linux-gnu-strip -DCMAKE_FIND_ROOT_PATH=powerpc64le-linux-gnu -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
          make -j $MAKEJ

          cd ../8bit
          ln -sf ../10bit/libx265.a libx265_main10.a
          ln -sf ../12bit/libx265.a libx265_main12.a
          $CMAKE ../../../source -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=ppc64le -DCMAKE_CXX_FLAGS="-m64" -DCMAKE_C_FLAGS="-m64" -DCMAKE_C_COMPILER=powerpc64le-linux-gnu-gcc -DCMAKE_CXX_COMPILER=powerpc64le-linux-gnu-g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_STRIP=powerpc64le-linux-gnu-strip -DCMAKE_FIND_ROOT_PATH=powerpc64le-linux-gnu -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF -DENABLE_CLI=OFF
        fi
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
ar -M <<EOF
CREATE libx265.a
ADDLIB libx265_main.a
ADDLIB libx265_main10.a
ADDLIB libx265_main12.a
SAVE
END
EOF
        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=generic-gnu
        else
          CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=generic-gnu
        fi
        make -j $MAKEJ
        make install
        cd ../libwebp-$WEBP_VERSION
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG .
        else
          CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=ppc64le -DCMAKE_CXX_FLAGS="-m64 -fPIC" -DCMAKE_C_FLAGS="-m64 -fPIC" -DCMAKE_C_COMPILER=powerpc64le-linux-gnu-gcc -DCMAKE_CXX_COMPILER=powerpc64le-linux-gnu-g++ -DCMAKE_STRIP=powerpc64le-linux-gnu-strip -DCMAKE_FIND_ROOT_PATH=powerpc64le-linux-gnu .
        fi
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --target=ppc64le-linux CFLAGS="-m64"
        else
          CC="powerpc64le-linux-gnu-gcc" CXX="powerpc64le-linux-gnu-g++" ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=powerpc64le-linux-gnu --build=ppc64le-linux CFLAGS="-m64"
        fi
        make -j $MAKEJ
        make install
        cd ../harfbuzz-$HARFBUZZ_VERSION
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
            meson setup build --prefix=$INSTALL_PATH $HARFBUZZ_CONFIG
        else
            echo "[binaries]" >> linux-ppc.ini
            echo "c = 'powerpc64le-linux-gnu-gcc'" >> linux-ppc.ini
            echo "cpp = 'powerpc64le-linux-gnu-g++'" >> linux-ppc.ini
            echo "ar = 'powerpc64le-linux-gnu-ar'" >> linux-ppc.ini
            echo "windres = 'powerpc64le-linux-gnu-windres'" >> linux-ppc.ini
            echo "strip = 'powerpc64le-linux-gnu-strip'" >> linux-ppc.ini
            echo "pkg-config = '/usr/bin/pkg-config'" >> linux-ppc.ini
            echo "" >> linux-ppc.ini
            echo "[host_machine]" >> linux-ppc.ini
            echo "system = 'linux'" >> linux-ppc.ini
            echo "cpu_family = 'ppc64'" >> linux-ppc.ini
            echo "cpu = 'ppc64'" >> linux-ppc.ini
            echo "endian = 'little'" >> linux-ppc.ini
            meson setup build --prefix=$INSTALL_PATH $HARFBUZZ_CONFIG --cross-file=linux-ppc.ini --pkg-config-path=/usr/bin/pkg-config
        fi
        cd build
        meson compile
        meson install
        cd ..
        cd ../nv-codec-headers-n$NVCODEC_VERSION
        make install PREFIX=$INSTALL_PATH
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBAOM_CONFIG ..
        else
          CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=ppc64le -DCMAKE_CXX_FLAGS="-m64 -fPIC" -DCMAKE_C_FLAGS="-m64 -fPIC" -DCMAKE_C_COMPILER=powerpc64le-linux-gnu-gcc -DCMAKE_CXX_COMPILER=powerpc64le-linux-gnu-g++ -DCMAKE_STRIP=powerpc64le-linux-gnu-strip -DCMAKE_FIND_ROOT_PATH=powerpc64le-linux-gnu $LIBAOM_CONFIG ..
        fi
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG ..
        else
          CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=ppc64le -DCMAKE_CXX_FLAGS="-m64 -fPIC" -DCMAKE_C_FLAGS="-m64 -fPIC" -DCMAKE_C_COMPILER=powerpc64le-linux-gnu-gcc -DCMAKE_CXX_COMPILER=powerpc64le-linux-gnu-g++ -DCMAKE_STRIP=powerpc64le-linux-gnu-strip -DCMAKE_FIND_ROOT_PATH=powerpc64le-linux-gnu $LIBSVTAV1_CONFIG ..
        fi
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          LDEXEFLAGS='-Wl,-rpath,\$$ORIGIN/' PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $ENABLE_VULKAN --enable-cuda --enable-cuvid --enable-nvenc --enable-pthreads --enable-libxcb --enable-libpulse --cc="gcc -m64" --extra-cflags="-I../include/ -I../include/libxml2 -I../include/mfx -I../include/svt-av1" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl -lz -lm" --disable-altivec || cat ffbuild/config.log
        else
          echo "configure ffmpeg cross compile"
          LDEXEFLAGS='-Wl,-rpath,\$$ORIGIN/' PKG_CONFIG_PATH=../lib/pkgconfig/:/usr/lib/powerpc64le-linux-gnu/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $ENABLE_VULKAN --enable-cuda --enable-cuvid --enable-nvenc --enable-pthreads --enable-libxcb --enable-libpulse --cc="powerpc64le-linux-gnu-gcc -m64" --extra-cflags="-I../include/ -I../include/libxml2 -I../include/mfx -I../include/svt-av1" --extra-ldflags="-L../lib/" --enable-cross-compile --target-os=linux --arch=ppc64le-linux --extra-libs="-lstdc++ -lpthread -ldl -lz -lm" --disable-altivec || cat ffbuild/config.log
        fi
        make -j $MAKEJ
        make install
        ;;

    macosx-arm64)
        export CFLAGS="-arch arm64 -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-apple-darwin
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        CC="clang -arch arm64 -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --host=aarch64-apple-darwin
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --host=aarch64-apple-darwin
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-apple-darwin
        make -j $MAKEJ V=0
        make install
        cd ../$OPUS
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-apple-darwin
        make -j $MAKEJ V=0
        make install
        cd ../$OPENCORE_AMR
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-apple-darwin
        make -j $MAKEJ V=0
        make install
        cd ../$VO_AMRWBENC
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-apple-darwin
        make -j $MAKEJ V=0
        make install
        cd ../$OPENSSL
        ./Configure darwin64-arm64-cc -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../srt-$LIBSRT_VERSION
        CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_C_COMPILER="clang" -DCMAKE_CXX_COMPILER="clang++" .

        make -j $MAKEJ V=0
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. OS=darwin ARCH=arm64 USE_ASM=No install-static CC="clang -arch arm64" CXX="clang++ -arch arm64"
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --disable-asm --disable-cli --host=aarch64-apple-darwin --extra-cflags="$CFLAGS"
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265/build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS -fPIC" -DCMAKE_C_FLAGS="$CFLAGS -fPIC" -DCMAKE_C_COMPILER="clang" -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
        make -j $MAKEJ

        cd ../10bit
        $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS -fPIC" -DCMAKE_C_FLAGS="$CFLAGS -fPIC" -DCMAKE_C_COMPILER="clang" -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF
        make -j $MAKEJ

        cd ../8bit
        ln -sf ../10bit/libx265.a libx265_main10.a
        ln -sf ../12bit/libx265.a libx265_main12.a
        $CMAKE ../../../source -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS -fPIC" -DCMAKE_C_FLAGS="$CFLAGS -fPIC" -DCMAKE_C_COMPILER="clang" -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_BUILD_TYPE=Release -DENABLE_ASSEMBLY=OFF -DENABLE_CLI=OFF
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
        /usr/bin/libtool -static -o libx265.a libx265_main.a libx265_main10.a libx265_main12.a 2>/dev/null

        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        sedinplace '/avx512/d' configure
        CC="clang -arch arm64" CXX="clang++ -arch arm64" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=generic-gnu
        make -j $MAKEJ
        sedinplace '/HAS_AVX512/d' vpx_dsp_rtcd.h
        make install
        cd ../libwebp-$WEBP_VERSION
        CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS -fPIC" -DCMAKE_C_FLAGS="$CFLAGS -fPIC" -DCMAKE_C_COMPILER="clang" -DCMAKE_CXX_COMPILER="clang++" .
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=aarch64-apple-darwin
        make -j $MAKEJ
        make install
        cd ../harfbuzz-$HARFBUZZ_VERSION
        echo "[binaries]" >> macos-arm.ini
        echo "c = 'clang'" >> macos-arm.ini
        echo "cpp = 'clang++'" >> macos-arm.ini
        echo "ar = 'ar'" >> macos-arm.ini
        echo "windres = 'windres'" >> macos-arm.ini
        echo "strip = 'strip'" >> macos-arm.ini
        echo "pkg-config='pkg-config'" >> macos-arm.ini
        echo "" >> macos-arm.ini
        echo "[properties]" >> macos-arm.ini
        echo "c_args = ['-arch', 'arm64']" >> macos-arm.ini
        echo "cpp_args = ['-arch', 'arm64']" >> macos-arm.ini
        echo "needs_exe_wrapper = true" >> macos-arm.ini
        echo "[host_machine]" >> macos-arm.ini
        echo "system = 'darwin'" >> macos-arm.ini
        echo "cpu_family = 'aarch64'" >> macos-arm.ini
        echo "cpu = 'arm64'" >> macos-arm.ini
        echo "endian = 'little'" >> macos-arm.ini
        meson setup build --prefix=$INSTALL_PATH $HARFBUZZ_CONFIG --cross-file=macos-arm.ini
        cd build
        meson compile
        meson install
        cd ..
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_CXX_FLAGS="$CXXFLAGS -fPIC" -DCMAKE_C_FLAGS="$CFLAGS -fPIC" -DCMAKE_C_COMPILER="clang" -DCMAKE_CXX_COMPILER="clang++" $LIBAOM_CONFIG -DAOM_ARCH_AARCH64=1 -DCONFIG_RUNTIME_CPU_DETECT:BOOL=OFF -DENABLE_NEON_I8MM=OFF ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG -DCOMPILE_C_ONLY=ON ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-macosx.patch
        LDEXEFLAGS='-Wl,-rpath,@loader_path/' PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $ENABLE_VULKAN --enable-pthreads --enable-indev=avfoundation --disable-libxcb --cc="clang -arch arm64" --extra-cflags="-I../include/ -I../include/libxml2 -I../include/mfx -I../include/svt-av1" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl -lz -lm" --enable-cross-compile --arch=arm64 --target-os=darwin || cat ffbuild/config.log
        make -j $MAKEJ
        make install
        ;;

    macosx-x86_64)
        export AS="nasm"
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        CC="clang -fPIC" ./configure --prefix=$INSTALL_PATH --static
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
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
        ./Configure darwin64-x86_64-cc -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../srt-$LIBSRT_VERSION
        CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar USE_ASM=No install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --disable-asm
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265/build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm
        make -j $MAKEJ

        cd ../10bit
        $CMAKE ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm
        make -j $MAKEJ

        cd ../8bit
        ln -sf ../10bit/libx265.a libx265_main10.a
        ln -sf ../12bit/libx265.a libx265_main12.a
        $CMAKE ../../../source -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
        /usr/bin/libtool -static -o libx265.a libx265_main.a libx265_main10.a libx265_main12.a 2>/dev/null

        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        sedinplace '/avx512/d' configure
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests
        make -j $MAKEJ
        sedinplace '/HAS_AVX512/d' vpx_dsp_rtcd.h
        make install
        cd ../libwebp-$WEBP_VERSION
        CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../harfbuzz-$HARFBUZZ_VERSION
        meson setup build --prefix=$INSTALL_PATH $HARFBUZZ_CONFIG
        cd build
        meson compile
        meson install
        cd ..
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBAOM_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        $CMAKE -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        patch -Np1 < ../../../ffmpeg-macosx.patch
        LDEXEFLAGS='-Wl,-rpath,@loader_path/' PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $ENABLE_VULKAN --enable-pthreads --enable-indev=avfoundation --disable-libxcb --extra-cflags="-I../include/ -I../include/libxml2 -I../include/mfx -I../include/svt-av1" --extra-ldflags="-L../lib/" --extra-libs="-lstdc++ -ldl -lz -lm" || cat ffbuild/config.log
        make -j $MAKEJ
        make install
        ;;

    windows-x86)
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        make -j $MAKEJ install -fwin32/Makefile.gcc BINARY_PATH=$INSTALL_PATH/bin/ INCLUDE_PATH=$INSTALL_PATH/include/ LIBRARY_PATH=$INSTALL_PATH/lib/
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --build=i686-w64-mingw32 CFLAGS="-m32 -msse2"
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --build=i686-w64-mingw32 CFLAGS="-m32 -msse2"
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=i686-w64-mingw32 CFLAGS="-m32"
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
        ./Configure mingw -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../srt-$LIBSRT_VERSION
        CC="gcc -m32" CXX="g++ -m32" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG -DENABLE_STDCXX_SYNC=ON .
        make -j $MAKEJ V=0
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86 USE_ASM=No install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=i686-w64-mingw32
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265/build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -G "MSYS Makefiles" ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm.exe
        make -j $MAKEJ

        cd ../10bit
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -G "MSYS Makefiles" ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm.exe
        make -j $MAKEJ

        cd ../8bit
        ln -sf ../10bit/libx265.a libx265_main10.a
        ln -sf ../12bit/libx265.a libx265_main12.a
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -G "MSYS Makefiles" ../../../source -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm.exe -DENABLE_CLI=OFF
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
ar -M <<EOF
CREATE libx265.a
ADDLIB libx265_main.a
ADDLIB libx265_main10.a
ADDLIB libx265_main12.a
SAVE
END
EOF
        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=x86-win32-gcc --disable-avx512
        make -j $MAKEJ
        make install
        cd ../libwebp-$WEBP_VERSION
        CC="gcc -m32" CXX="g++ -m32" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=i686-w64-mingw32 CFLAGS="-m32"
        make -j $MAKEJ
        make install
        cd ../mfx_dispatch-$MFX_VERSION
        sedinplace 's:${SOURCES}:${SOURCES} src/mfx_driver_store_loader.cpp:g' CMakeLists.txt
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release .
        make -j $MAKEJ
        make install
        cd ../nv-codec-headers-n$NVCODEC_VERSION
        make install PREFIX=$INSTALL_PATH
        cd ../harfbuzz-$HARFBUZZ_VERSION
        echo "[binaries]" >> windows-native.ini
        echo "c = 'gcc'" >> windows-native.ini
        echo "cpp = 'g++'" >> windows-native.ini
        echo "ar = 'ar'" >> windows-native.ini
        echo "windres = 'windres'" >> windows-native.ini
        echo "strip = 'strip'" >> windows-native.ini
        meson setup build --prefix=$INSTALL_PATH $HARFBUZZ_CONFIG --native-file windows-native.ini -Dcpp_args="-m32" -Dc_args="-m32"
        cd build
        meson compile
        meson install
        cd ..
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        CC="gcc -m32" CXX="g++ -m32" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBAOM_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        CC="gcc -m32" CXX="g++ -m32" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $ENABLE_VULKAN --enable-cuda --enable-cuvid --enable-nvenc --enable-libmfx --enable-w32threads --enable-indev=dshow --target-os=mingw32 --cc="gcc -m32" --extra-cflags="-DLIBXML_STATIC -I../include/ -I../include/libxml2 -I../include/mfx/ -I../include/svt-av1" --extra-ldflags="-L../lib/" --extra-libs="-static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lgcc_eh -lWs2_32 -lcrypt32 -lpthread -lz -lm -Wl,-Bdynamic -lole32 -luuid" || cat ffbuild/config.log
        make -j $MAKEJ
        make install
        ;;

    windows-x86_64)
        echo ""
        echo "--------------------"
        echo "Building zimg"
        echo "--------------------"
        echo ""
        cd zimg-release-$ZIMG_VERSION
        autoreconf -iv
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building zlib"
        echo "--------------------"
        echo ""
        cd ../$ZLIB
        make -j $MAKEJ install -fwin32/Makefile.gcc BINARY_PATH=$INSTALL_PATH/bin/ INCLUDE_PATH=$INSTALL_PATH/include/ LIBRARY_PATH=$INSTALL_PATH/lib/
        echo ""
        echo "--------------------"
        echo "Building LAME"
        echo "--------------------"
        echo ""
        cd ../$LAME
        ./configure --prefix=$INSTALL_PATH --disable-frontend --disable-shared --with-pic --build=x86_64-w64-mingw32 CFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building XML2"
        echo "--------------------"
        echo ""
        cd ../$XML2
        ./configure --prefix=$INSTALL_PATH $LIBXML_CONFIG --build=x86_64-w64-mingw32 CFLAGS="-m64"
        make -j $MAKEJ V=0
        make install
        echo ""
        echo "--------------------"
        echo "Building speex"
        echo "--------------------"
        echo ""
        cd ../$SPEEX
        PKG_CONFIG= ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --build=x86_64-w64-mingw32 CFLAGS="-m64"
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
        ./Configure mingw64 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../srt-$LIBSRT_VERSION
        CC="gcc -m64" CXX="g++ -m64" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $SRT_CONFIG -DENABLE_STDCXX_SYNC=ON .
        make -j $MAKEJ V=0
        make install
        cd ../openh264-$OPENH264_VERSION
        make -j $MAKEJ DESTDIR=./ PREFIX=.. AR=ar ARCH=x86_64 USE_ASM=No install-static
        cd ../$X264
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-opencl --host=x86_64-w64-mingw32
        make -j $MAKEJ V=0
        make install
        cd ../x265-$X265/build/linux
        # from x265 multilib.sh
        mkdir -p 8bit 10bit 12bit

        cd 12bit
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -G "MSYS Makefiles" ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm.exe
        make -j $MAKEJ

        cd ../10bit
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -G "MSYS Makefiles" ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm.exe
        make -j $MAKEJ

        cd ../8bit
        ln -sf ../10bit/libx265.a libx265_main10.a
        ln -sf ../12bit/libx265.a libx265_main12.a
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -G "MSYS Makefiles" ../../../source -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DENABLE_SHARED:BOOL=OFF -DENABLE_LIBNUMA=OFF -DCMAKE_BUILD_TYPE=Release -DNASM_EXECUTABLE:FILEPATH=$INSTALL_PATH/bin/nasm.exe -DENABLE_CLI=OFF
        make -j $MAKEJ

        # rename the 8bit library, then combine all three into libx265.a
        mv libx265.a libx265_main.a
ar -M <<EOF
CREATE libx265.a
ADDLIB libx265_main.a
ADDLIB libx265_main10.a
ADDLIB libx265_main12.a
SAVE
END
EOF
        make install
        # ----
        cd ../../../
        cd ../libvpx-$VPX_VERSION
        ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-examples --disable-unit-tests --target=x86_64-win64-gcc --disable-avx512
        make -j $MAKEJ
        make install
        cd ../libwebp-$WEBP_VERSION
        CC="gcc -m64" CXX="g++ -m64" CFLAGS="-I$INSTALL_PATH/include/" CXXFLAGS="-I$INSTALL_PATH/include/" LDFLAGS="-L$INSTALL_PATH/lib/" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $WEBP_CONFIG .
        make -j $MAKEJ V=0
        make install
        cd ../freetype-$FREETYPE_VERSION
        ./configure --prefix=$INSTALL_PATH --with-bzip2=no --with-harfbuzz=no --with-png=no --with-brotli=no --enable-static --disable-shared --with-pic --host=x86_64-w64-mingw32 CFLAGS="-m64"
        make -j $MAKEJ
        make install
        cd ../mfx_dispatch-$MFX_VERSION
        sedinplace 's:${SOURCES}:${SOURCES} src/mfx_driver_store_loader.cpp:g' CMakeLists.txt
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release .
        make -j $MAKEJ
        make install
        cd ../nv-codec-headers-n$NVCODEC_VERSION
        make install PREFIX=$INSTALL_PATH
        cd ../harfbuzz-$HARFBUZZ_VERSION
        echo "[binaries]" >> windows-native.ini
        echo "c = 'gcc'" >> windows-native.ini
        echo "cpp = 'g++'" >> windows-native.ini
        echo "ar = 'ar'" >> windows-native.ini
        echo "windres = 'windres'" >> windows-native.ini
        echo "strip = 'strip'" >> windows-native.ini
        meson setup build --prefix=$INSTALL_PATH $HARFBUZZ_CONFIG --native-file windows-native.ini -Dcpp_args="-m64" -Dc_args="-m64"
        cd build
        meson compile
        meson install
        cd ..
        cd ../libaom-$AOMAV1_VERSION
        mkdir -p build_release
        cd build_release
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBAOM_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../SVT-AV1-v$SVTAV1_VERSION
        mkdir -p build_release
        cd build_release
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH $LIBSVTAV1_CONFIG ..
        make -j $MAKEJ
        make install
        cd ..
        cd ../ffmpeg-$FFMPEG_VERSION
        PKG_CONFIG_PATH=../lib/pkgconfig/ ./configure --prefix=.. $DISABLE $ENABLE $ENABLE_VULKAN --enable-cuda --enable-cuvid --enable-nvenc --enable-libmfx --enable-w32threads --enable-indev=dshow --target-os=mingw32 --cc="gcc -m64" --extra-cflags="-DLIBXML_STATIC -I../include/ -I../include/libxml2 -I../include/mfx/ -I../include/svt-av1" --extra-ldflags="-L../lib/" --extra-libs="-static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lgcc_eh -lWs2_32 -lcrypt32 -lpthread -lz -lm -Wl,-Bdynamic -lole32 -luuid" || cat ffbuild/config.log
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
