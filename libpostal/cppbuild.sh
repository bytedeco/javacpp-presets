#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libpostal
    popd
    exit
fi

LIBPOSTAL_VERSION=1.1-alpha
download https://github.com/openvenues/libpostal/archive/v$LIBPOSTAL_VERSION.tar.gz libpostal-$LIBPOSTAL_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=$(pwd)

echo "Decompressing archives..."
tar --totals -xf ../libpostal-$LIBPOSTAL_VERSION.tar.gz
cd libpostal-*

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
        SOURCE_PATH=$(pwd)
        CC="gcc -m64 -Duint=int -static-libgcc"
        bash -lc "cd $SOURCE_PATH && ./bootstrap.sh"
        bash -lc "cd $SOURCE_PATH && ./configure --prefix=$INSTALL_PATH --disable-data-download"
        #bash -lc "cd $SOURCE_PATH && ./configure --prefix=$INSTALL_PATH --datadir=/c/[...some dir with a few GB of space...]"
        bash -lc "cd $SOURCE_PATH && make -j $MAKEJ V=0"
        bash -lc "cd $SOURCE_PATH && make install"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
;;

esac

cd ../..
