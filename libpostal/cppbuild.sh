#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libpostal
    popd
    exit
fi

LIBPOSTAL_VERSION=1.1
download https://github.com/openvenues/libpostal/archive/v$LIBPOSTAL_VERSION.tar.gz libpostal-$LIBPOSTAL_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=$(pwd)

echo "Decompressing archives..."
tar --totals -xf ../libpostal-$LIBPOSTAL_VERSION.tar.gz
cd libpostal-$LIBPOSTAL_VERSION
if [[ "${ACLOCAL_PATH:-}" == C:\\msys64\\* ]]; then
    export ACLOCAL_PATH=/mingw64/share/aclocal:/usr/share/aclocal
fi

# Work around build issues on Mac and Windows
sedinplace 's/-Werror=format-security/-Wno-implicit-function-declaration/g' src/Makefile.am
sedinplace '/_rand48_/d' src/klib/drand48.h

case $PLATFORM in
    linux-x86_64)
        ./bootstrap.sh
        ./configure --prefix=$INSTALL_PATH --disable-data-download
        #./configure --prefix=$INSTALL_PATH --datadir=[...some dir with a few GB of space...]
        make -j $MAKEJ V=0
        make install
        ;;
    macosx-*)
        ./bootstrap.sh
        sed -i="" 's/-install_name \\$rpath/-install_name @rpath/g' configure
        ./configure --prefix=$INSTALL_PATH --disable-data-download
        #./configure --prefix=$INSTALL_PATH --datadir=[...some dir with a few GB of space...]
        make -j $MAKEJ V=0
        make install
        ;;
    windows-x86_64)
        cp -rf windows/* ./
        CC="gcc -m64 -Duint=int -static-libgcc"
        ./bootstrap.sh
        ./configure --prefix=$INSTALL_PATH --disable-data-download
        #./configure --prefix=$INSTALL_PATH --datadir=/c/[...some dir with a few GB of space...]
        make -j $MAKEJ V=0
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
;;

esac

cd ../..
