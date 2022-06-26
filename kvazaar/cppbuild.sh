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
    linux-x86_64)
        CC="gcc -m64 -fPIC" ./configure --prefix=$INSTALL_PATH --enable-static --enable-pic --disable-cli --disable-shared
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..