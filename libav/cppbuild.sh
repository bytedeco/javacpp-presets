if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

LIBAV_DIR=libav-master

download https://github.com/libav/libav/archive/master.tar.gz libav-master.tar.gz
download ftp://ftp.videolan.org/pub/videolan/x264/snapshots/last_stable_x264.tar.bz2 last_stable_x264.tar.bz2
download https://github.com/lu-zero/speex/archive/master.tar.gz speex-master.tar.gz

tar -xzvf libav-master.tar.gz

tar -xjvf last_stable_x264.tar.bz2

X264=`echo x264-snapshot-*`

tar -xzvf speex-master.tar.gz

mkdir -p $PLATFORM

SPEEX="speex-master"

DEPS_PATH="${PWD}/deps"
INST_PATH="${PWD}/${PLATFORM}"

export PKG_CONFIG_LIBDIR="${DEPS_PATH}/lib/pkgconfig"
export PKG_CONFIG_SYSROOT_DIR="${DEPS_PATH}"
export PATH="$PATH:${ANDROID_BINDIR}"

case $PLATFORM in
    android-arm)
        pushd $SPEEX
        autoreconf -i
        ./configure --prefix="/" --disable-shared --enable-static --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" CFLAGS="--sysroot=$ANDROID_ROOT" CPPFLAGS="--sysroot=$ANDROID_ROOT" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc" --without-ogg
        make -j4
        make DESTDIR="${DEPS_PATH}" install
        popd
        pushd $X264
        ./configure --prefix="/" --enable-static --enable-pic --disable-cli --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --host=arm-linux --extra-ldflags="-nostdlib -Wl,--fix-cortex-a8 -lgcc -ldl -lz -lm -lc"
        make -j4
        make DESTDIR="${DEPS_PATH}" install
        popd
        pushd $LIBAV_DIR
        ./configure --prefix="${INST_PATH}" --enable-shared --enable-gpl --enable-version3 --enable-runtime-cpudetect --disable-outdev=sdl --enable-libx264 --enable-libspeex --enable-cross-compile --cross-prefix="$ANDROID_BIN-" --sysroot="$ANDROID_ROOT" --target-os=android --arch=arm --cpu=armv7-a --extra-ldflags="-Wl,--fix-cortex-a8 -lgcc -ldl -lz -lm -lc -L${DEPS_PATH}/lib" --extra-cflags="-I${DEPS_PATH}/include" --disable-programs --pkg-config=pkg-config
        make -j4
        make install
        popd
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported yet"
        ;;
esac
