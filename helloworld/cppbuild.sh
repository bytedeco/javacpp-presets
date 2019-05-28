#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" helloworld
    popd
    exit
fi

HELLOWORLD_VERSION=master
download https://github.com/matteodg/helloworld/archive/$HELLOWORLD_VERSION.tar.gz helloworld-$HELLOWORLD_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../helloworld-$HELLOWORLD_VERSION.tar.gz
cd helloworld-$HELLOWORLD_VERSION

mkdir -p build
cd build
case $PLATFORM in
    linux-x86)
        gcc -m32 -fPIC ../helloworld.c -shared -o libhelloworld.so
        mkdir -p ../../include
        cp ../helloworld.h ../../include
        mkdir -p ../../lib
        cp libhelloworld.so ../../lib
        ;;
    linux-x86_64)
        gcc -m64 -fPIC ../helloworld.c -shared -o libhelloworld.so
        mkdir -p ../../include
        cp ../helloworld.h ../../include
        mkdir -p ../../lib
        cp libhelloworld.so ../../lib
        ;;
    macosx-*)
        clang -fPIC ../helloworld.c -shared -o libhelloworld.dylib
        mkdir -p ../../include
        cp ../helloworld.h ../../include
        mkdir -p ../../lib
        cp libhelloworld.dylib ../../lib
        ;;
    windows-x86)
        gcc -m32 -fPIC ../helloworld.c -shared -o helloworld.dll
        mkdir -p ../../include
        cp ../helloworld.h ../../include
        mkdir -p ../../lib
        cp helloworld.dll ../../lib
        ;;
    windows-x86_64)
        gcc -m64 -fPIC ../helloworld.c -shared -o helloworld.dll
        mkdir -p ../../include
        cp ../helloworld.h ../../include
        mkdir -p ../../lib
        cp helloworld.dll ../../lib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
