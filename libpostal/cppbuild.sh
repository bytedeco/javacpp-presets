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
tar -xzvf ../libpostal-$LIBPOSTAL_SHA.tar.gz
cd libpostal-*

case $PLATFORM in
    linux-x86_64)
    ./bootstrap.sh
    #./configure --datadir=[...some dir with a few GB of space...]
    ./configure --disable-data-download
    make -j4
    make install
    mkdir -p ../lib
    mkdir -p ../include
    cp src/.libs/libpostal.so ../lib/liblibpostal.so
    cp src/libpostal.h ../include
        ;;
    macosx-*)
    ./bootstrap.sh
    #./configure --datadir=[...some dir with a few GB of space...]
    ./configure --disable-data-download
    make -j4
    make install
    mkdir -p ../lib
    mkdir -p ../include
    cp src/.libs/libpostal.dylib ../lib/liblibpostal.dylib
    cp src/libpostal.h ../include
        ;;
    windows-x86_64)
    cp -rf windows/* ./
    cwd=$(pwd)
    bash -lc "cd $cwd && autoreconf -fi --warning=no-portability"
    #bash -lc "cd $cwd && ./configure --datadir=/c" #[...some dir with a few GB of space...]
    bash -lc "cd $cwd && ./configure --disable-data-download"
    bash -lc "cd $cwd && make -j4"
    bash -lc "cd $cwd && make install"
    lib.exe /def:libpostal.def /out:libpostal.lib /machine:x64
    mkdir -p ../bin
    mkdir -p ../lib
    mkdir -p ../include
    cp src/.libs/libpostal-1.dll ../bin/libpostal.dll
    cp libpostal.lib ../lib
    cp src/libpostal.h ../include
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
;;

esac

cd ../..