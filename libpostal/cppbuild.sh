#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libpostal
    popd
    exit
fi

LIBPOSTAL_SHA=1e7cc23
download https://github.com/openvenues/libpostal/archive/$LIBPOSTAL_SHA.tar.gz libpostal-$LIBPOSTAL_SHA.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=$(pwd)
tar -xzvf ../libpostal-$LIBPOSTAL_SHA.tar.gz
cd libpostal-*

case $PLATFORM in
    linux-x86_64)
    ./bootstrap.sh
    ./configure --prefix=$INSTALL_PATH --disable-data-download
    #./configure --prefix=$INSTALL_PATH --datadir=[...some dir with a few GB of space...]
    make -j4
    make install
        ;;
    macosx-*)
    ./bootstrap.sh
    ./configure --prefix=$INSTALL_PATH --disable-data-download
    #./configure --prefix=$INSTALL_PATH --datadir=[...some dir with a few GB of space...]
    make -j4
    make install
        ;;
    windows-x86_64)
    cp -rf windows/* ./
    SOURCE_PATH=$(pwd)
    CC="gcc -m64 -Duint=int -static-libgcc"
    bash -lc "cd $SOURCE_PATH && ./bootstrap.sh"
    bash -lc "cd $SOURCE_PATH && ./configure --prefix=$INSTALL_PATH --disable-data-download"
    #bash -lc "cd $SOURCE_PATH && ./configure --prefix=$INSTALL_PATH --datadir=/c/[...some dir with a few GB of space...]"
    bash -lc "cd $SOURCE_PATH && make -j4"
    bash -lc "cd $SOURCE_PATH && make install"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
;;

esac

cd ../..