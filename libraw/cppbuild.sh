#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libraw
    popd
    exit
fi

# Compilation instructions at https://www.libraw.org/docs/Install-LibRaw.html

LIBRAW_VERSION=0.21.1
download https://github.com/LibRaw/LibRaw/archive/refs/tags/$LIBRAW_VERSION.zip LibRaw-$LIBRAW_VERSION.zip
unzip -o LibRaw-$LIBRAW_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include/libraw lib
unzip -o ../LibRaw-$LIBRAW_VERSION.zip
cd LibRaw-$LIBRAW_VERSION

# TODO add: zlib, libjasper, libjpeg8

case $PLATFORM in
    linux-x86_64)
        autoreconf --install
        ./configure --prefix=$INSTALL_PATH --disable-examples --disable-static --with-pic
        make -j $MAKEJ
        make install
        ;;
    macosx-arm64)
        export CC="clang -arch arm64"
        export CXX="clang++ -arch arm64"
        autoreconf --install
        ./configure --prefix=$INSTALL_PATH --disable-examples --enable-static --with-pic --host="aarch64-apple-darwin"
        make -j $MAKEJ
        make install
        ;;
    macosx-x86_64)
        autoreconf --install
        ./configure --prefix=$INSTALL_PATH --disable-examples --enable-static --with-pic
        make -j $MAKEJ
        make install
        ;;
    windows-x86_64)
        nmake -f Makefile.msvc
        cp libraw/*.h ../include/libraw
        cp lib/*.lib ../lib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
