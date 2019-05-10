#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" helloworld
    popd
    exit
fi

HELLOWORLD_VERSION=1.0

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

if [[ -e "helloworld-$HELLOWORLD_VERSION" ]]; then
    cd helloworld-$HELLOWORLD_VERSION
    git fetch
    git pull
else
#    git clone https://github.com/matteodg/helloworld.git -b $HELLOWORLD_VERSION helloworld-$HELLOWORLD_VERSION
    git clone https://github.com/matteodg/helloworld.git -b master helloworld-$HELLOWORLD_VERSION
    cd helloworld-$HELLOWORLD_VERSION
fi


case $PLATFORM in
    linux-x86)
        mkdir -p build
        cd build
        gcc -m32 ../helloworld.c -shared -o libhelloworld.so
        mkdir -p ../../include
        cp ../helloworld.h ../../include
        mkdir -p ../../lib
        cp libhelloworld.so ../../lib
        ;;
    linux-x86_64)
        mkdir -p build
        cd build
        gcc ../helloworld.c -shared -o libhelloworld.so
        mkdir -p ../../include
        cp ../helloworld.h ../../include
        mkdir -p ../../lib
        cp libhelloworld.so ../../lib
        ;;
    windows-x86)
        mkdir -p build
        cd build
        gcc -m32 ../helloworld.c -shared -o libhelloworld.dll
        mkdir -p ../../include
        cp ../helloworld.h ../../include
        mkdir -p ../../lib
        cp libhelloworld.dll ../../lib
        ;;
    windows-x86_64)
        mkdir -p build
        cd build
        gcc -m64 ../helloworld.c -shared -o libhelloworld.dll
        mkdir -p ../../include
        cp ../helloworld.h ../../include
        mkdir -p ../../lib
        cp libhelloworld.dll ../../lib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
