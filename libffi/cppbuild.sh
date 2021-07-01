#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libffi
    popd
    exit
fi

LIBFFI_VERSION=3.4.2
download https://github.com/libffi/libffi/releases/download/v$LIBFFI_VERSION/libffi-$LIBFFI_VERSION.tar.gz libffi-$LIBFFI_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../libffi-$LIBFFI_VERSION.tar.gz
cd libffi-$LIBFFI_VERSION

case $PLATFORM in
    android-arm)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="$ANDROID_LIBS"
        ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install-strip
        ;;
    android-arm64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="$ANDROID_LIBS"
        ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory --host="aarch64-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install-strip
        ;;
    android-x86)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="$ANDROID_LIBS"
        ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install-strip
        ;;
    android-x86_64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="$ANDROID_LIBS"
        ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory --host="x86_64-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86)
        CC="gcc -m32" ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        CC="gcc -m64" ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory
        make -j $MAKEJ
        make install-strip
        ;;
    linux-armhf)
        CC="arm-linux-gnueabihf-gcc" ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory --host="arm-linux-gnueabihf"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-arm64)
        CC="aarch64-linux-gnu-gcc" ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory --host="aarch64-linux-gnu"
        make -j $MAKEJ
        make install-strip
        ;;
    linux-ppc64le)
        CC="powerpc64le-linux-gnu-gcc" ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory --host="powerpc64le-linux-gnu"
        make -j $MAKEJ
        make install-strip
        ;;
    macosx-*)
        sedinplace 's/\\\$rpath/@rpath/g' configure
        CC="clang" ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        CC="../msvcc.sh -m32" LD="link" CPP="cl -nologo -EP" CPPFLAGS="-DFFI_BUILDING_DLL" ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory
        make -j $MAKEJ
        make install || true
        cp */.libs/* ../lib
        ;;
    windows-x86_64)
        CC="../msvcc.sh -m64" LD="link" CPP="cl -nologo -EP" CPPFLAGS="-DFFI_BUILDING_DLL" ./configure --prefix="$INSTALL_PATH" --disable-multi-os-directory
        make -j $MAKEJ
        make install || true
        cp */.libs/* ../lib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
