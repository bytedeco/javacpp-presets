#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libecl
    popd
    exit
fi

LIBECL_VERSION=2.9.1
download https://github.com/equinor/libecl/archive/$LIBECL_VERSION.zip libecl-$LIBECL_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
unzip -o ../libecl-$LIBECL_VERSION.zip


cd libecl-$LIBECL_VERSION

mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../ -DENABLE_PYTHON=OFF

case $PLATFORM in
    linux-x86)
        make
        make install
        ;;
    linux-x86_64)
        make
        make install
        ;;
    macosx-*)
        make
        make install
        ;;
    windows-x86)
        cmake --build . --config Release --target install
        ;;
    windows-x86_64)
        cmake --build . --config Release --target install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..

