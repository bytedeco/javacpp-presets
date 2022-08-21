#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" kvazaar
    popd
    exit
fi

KVAZAAR_VERSION=2.1.0
echo "Downloading archives if needed..."
download https://github.com/ultravideo/kvazaar/releases/download/v$KVAZAAR_VERSION/kvazaar-$KVAZAAR_VERSION.tar.gz kvazaar-$KVAZAAR_VERSION.tar.gz
mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar -xzvf ../kvazaar-$KVAZAAR_VERSION.tar.gz
cd kvazaar-$KVAZAAR_VERSION

case $PLATFORM in
    linux-arm64)
        CC="aarch64-linux-gnu-gcc" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-shared --host=aarch64-linux-gnu
        make -j $MAKEJ
        make install
        ;;
    linux-armhf)
        CC="arm-linux-gnueabihf-gcc" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-shared --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        sed -i s/elf64ppc/elf64lppc/ configure
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-shared
        else
          CC="powerpc64le-linux-gnu-gcc -m64" ./configure --host=powerpc64le-linux-gnu --build=ppc64le-linux --prefix=$INSTALL_PATH --enable-static --enable-pic  --disable-shared 
        fi
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        CC="gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-shared
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        ./configure --prefix=$INSTALL_PATH --enable-static --disable-shared
        make -j $MAKEJ
        make install
        ;;
    windows-x86_64)
        CC="gcc -m64 -Duint=int -static-libgcc" ./configure --prefix=$INSTALL_PATH --disable-shared
        make -j $MAKEJ
        make install
        ls -lR ..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..