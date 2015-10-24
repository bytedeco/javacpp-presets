#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" llvm
    popd
    exit
fi

LLVM_VERSION=3.7.0
download http://llvm.org/releases/$LLVM_VERSION/llvm-$LLVM_VERSION.src.tar.xz llvm-$LLVM_VERSION.src.tar.xz
download http://llvm.org/releases/$LLVM_VERSION/cfe-$LLVM_VERSION.src.tar.xz cfe-$LLVM_VERSION.src.tar.xz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
xz -dc ../llvm-$LLVM_VERSION.src.tar.xz | tar -xvf -
cd llvm-$LLVM_VERSION.src
mkdir -p build tools
cd tools
xz -dc ../../../cfe-$LLVM_VERSION.src.tar.xz | tar -xvf -
rm -Rf clang
mv cfe-$LLVM_VERSION.src clang
cd ../build

case $PLATFORM in
    linux-x86)
        ../configure --prefix=$INSTALL_PATH --enable-shared --enable-optimized CC="clang -m32" CXX="clang++ -m32"
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        ../configure --prefix=$INSTALL_PATH --enable-shared --enable-optimized CC="clang -m64" CXX="clang++ -m64"
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        ../configure --prefix=$INSTALL_PATH --enable-shared --enable-optimized
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
