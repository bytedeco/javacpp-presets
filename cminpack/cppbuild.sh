#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" cminpack
    popd
    exit
fi

CMINPACK_VERSION=1.3.8
download https://github.com/devernay/cminpack/archive/v$CMINPACK_VERSION.tar.gz cminpack-$CMINPACK_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib bin
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../cminpack-$CMINPACK_VERSION.tar.gz

cd cminpack-$CMINPACK_VERSION
patch -Np1 < ../../../cminpack.patch

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

export C_INCLUDE_PATH="$OPENBLAS_PATH/include"
export LDFLAGS="-L$OPENBLAS_PATH/ -L$OPENBLAS_PATH/lib/"
export LD_LIBRARY_PATH=$OPENBLAS_PATH/:$OPENBLAS_PATH/lib/

case $PLATFORM in
    linux-x86)
        make -j $MAKEJ CC="gcc -m32 -fPIC" double
        make -j $MAKEJ CC="gcc -m32 -fPIC" lapack
        make -j $MAKEJ CC="gcc -m32 -fPIC" float
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=  install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=l install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=s install
        ;;
    linux-x86_64)
        make -j $MAKEJ CC="gcc -m64 -fPIC" double
        make -j $MAKEJ CC="gcc -m64 -fPIC" lapack
        make -j $MAKEJ CC="gcc -m64 -fPIC" float
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=  install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=l install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=s install
        ;;
    linux-armhf)
        make -j $MAKEJ CC="arm-linux-gnueabihf-gcc -fPIC" double
        make -j $MAKEJ CC="arm-linux-gnueabihf-gcc -fPIC" lapack
        make -j $MAKEJ CC="arm-linux-gnueabihf-gcc -fPIC" float
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=  install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=l install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=s install
        ;;
    linux-arm64)
        make -j $MAKEJ CC="aarch64-linux-gnu-gcc -fPIC" double
        make -j $MAKEJ CC="aarch64-linux-gnu-gcc -fPIC" lapack
        make -j $MAKEJ CC="aarch64-linux-gnu-gcc -fPIC" float
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=  install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=l install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=s install
        ;;
    linux-ppc64le)
        make -j $MAKEJ CC="powerpc64le-linux-gnu-gcc -m64 -fPIC" double
        make -j $MAKEJ CC="powerpc64le-linux-gnu-gcc -m64 -fPIC" lapack
        make -j $MAKEJ CC="powerpc64le-linux-gnu-gcc -m64 -fPIC" float
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=  install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=l install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=s install
        ;;
    macosx-*)
        export CC="$(ls -1 /usr/local/bin/gcc-? | head -n 1)"
        make -j $MAKEJ CC="$CC -fPIC" double
        make -j $MAKEJ CC="$CC -fPIC" lapack
        make -j $MAKEJ CC="$CC -fPIC" float
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=  install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=l install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=s install
        ;;
    windows-x86)
        make -j $MAKEJ CC="gcc -m32 -fPIC" double
        make -j $MAKEJ CC="gcc -m32 -fPIC" lapack
        make -j $MAKEJ CC="gcc -m32 -fPIC" float
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=  install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=l install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=s install
        ;;
    windows-x86_64)
        make -j $MAKEJ CC="gcc -m64 -fPIC" double
        make -j $MAKEJ CC="gcc -m64 -fPIC" lapack
        make -j $MAKEJ CC="gcc -m64 -fPIC" float
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=  install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=l install
        make DESTDIR="$INSTALL_PATH" LIBSUFFIX=s install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

unset C_INCLUDE_PATH
unset LDFLAGS
unset LD_LIBRARY_PATH

cd ../..
