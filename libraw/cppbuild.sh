#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libraw
    popd
    exit
fi

LIBRAW_VERSION=0.18.8
download https://www.libraw.org/data/LibRaw-$LIBRAW_VERSION.tar.gz LibRaw-$LIBRAW_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
echo "Decompressing archives..."
tar --totals -xzf ../LibRaw-$LIBRAW_VERSION.tar.gz
cd LibRaw-$LIBRAW_VERSION

case $PLATFORM in
    linux-x86_64)
        autoreconf -fi --warnings=portability
        sed -i="" 's/-install_name \\$rpath/-install_name @rpath/g' configure
        ./configure --prefix=$(pwd)/.. --enable-static
        make -j $MAKEJ
        make install
        ;;
    macosx-x86_64)
        autoreconf -fi --warnings=portability
        sed -i="" 's/-install_name \\$rpath/-install_name @rpath/g' configure
        ./configure --prefix=$(pwd)/.. --enable-static
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
