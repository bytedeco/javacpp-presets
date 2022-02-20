#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" gsl
    popd
    exit
fi

GSL_VERSION=2.7
download http://ftp.gnu.org/gnu/gsl/gsl-$GSL_VERSION.tar.gz gsl-$GSL_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -f "$P/include/openblas_config.h" ]]; then
            OPENBLAS_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"

echo "Decompressing archives..."
tar --totals -xzf ../gsl-$GSL_VERSION.tar.gz
cd gsl-$GSL_VERSION

export GSL_LDFLAGS="-L$OPENBLAS_PATH/ -L$OPENBLAS_PATH/lib/ -lopenblas"
case $PLATFORM in
    android-arm)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="-ldl -lm -lc"
        patch -Np1 < ../../../gsl-android.patch
        ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ V=0
        make install-strip
        ;;
    android-arm64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="-ldl -lm -lc"
        patch -Np1 < ../../../gsl-android.patch
        ./configure --prefix=$INSTALL_PATH --host="aarch64-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ V=0
        make install-strip
        ;;
     android-x86)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="-ldl -lm -lc"
        patch -Np1 < ../../../gsl-android.patch
        ./configure --prefix=$INSTALL_PATH --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ V=0
        make install-strip
        ;;
     android-x86_64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export CC="$ANDROID_CC $ANDROID_FLAGS"
        export STRIP="$ANDROID_PREFIX-strip"
        export LIBS="-ldl -lm -lc"
        patch -Np1 < ../../../gsl-android.patch
        ./configure --prefix=$INSTALL_PATH --host="x86_64-linux-android" --with-sysroot="$ANDROID_ROOT"
        make -j $MAKEJ V=0
        make install-strip
        ;;
    linux-x86)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m32"
        make -j $MAKEJ V=0
        make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m64"
        make -j $MAKEJ V=0
        make install-strip
        ;;
    linux-armhf)
        export LDFLAGS="-Wl,-rpath-link,$(dirname $(dirname $(which arm-linux-gnueabihf-gcc)))/arm-linux-gnueabihf/lib/"
        ./configure --prefix=$INSTALL_PATH --host=arm-linux-gnueabihf
        make -j $MAKEJ V=0
        make install-strip
        ;;
    linux-arm64)
        ./configure --prefix=$INSTALL_PATH --host=aarch64-linux-gnu
        make -j $MAKEJ V=0
        make install-strip
        ;;
    linux-ppc64le)
        sed -i s/elf64ppc/elf64lppc/ configure
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH CC="gcc -m64"
        else
          ./configure --prefix=$INSTALL_PATH --host=powerpc64le-linux-gnu --build=ppc64le-linux CC="powerpc64le-linux-gnu-gcc -m64"
        fi
        make -j $MAKEJ V=0
        make install-strip
        ;;
    macosx-*)
        #patch -Np1 < ../../../gsl-macosx.patch
        sedinplace 's/\\\$rpath/@rpath/g' aclocal.m4 configure
        ./configure --prefix=$INSTALL_PATH
        make -j $MAKEJ V=0
        make install-strip
        ;;
    windows-x86)
        patch -Np1 < ../../../gsl-windows.patch
        ./configure --prefix=$INSTALL_PATH CC="gcc -m32 -static-libgcc"
        make -j $MAKEJ V=0
        make install-strip
        ;;
    windows-x86_64)
        patch -Np1 < ../../../gsl-windows.patch
        ./configure --prefix=$INSTALL_PATH CC="gcc -m64 -static-libgcc"
        make -j $MAKEJ V=0
        make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
