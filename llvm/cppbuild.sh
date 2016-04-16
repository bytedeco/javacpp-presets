#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" llvm
    popd
    exit
fi

case $PLATFORM in
    linux-x86)
        export CC="clang -m32"
        export CXX="clang++ -m32"
        ;;
    linux-x86_64)
        export CC="clang -m64"
        export CXX="clang++ -m64"
        ;;
    macosx-*)
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

LLVM_VERSION=3.8.0
download http://llvm.org/releases/$LLVM_VERSION/llvm-$LLVM_VERSION.src.tar.xz llvm-$LLVM_VERSION.src.tar.xz
download http://llvm.org/releases/$LLVM_VERSION/cfe-$LLVM_VERSION.src.tar.xz cfe-$LLVM_VERSION.src.tar.xz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
tar xvf ../llvm-$LLVM_VERSION.src.tar.xz
cd llvm-$LLVM_VERSION.src
mkdir -p build tools
cd tools
tar xvf ../../../cfe-$LLVM_VERSION.src.tar.xz
rm -Rf clang
mv cfe-$LLVM_VERSION.src clang
cd ../build

../configure --prefix=$INSTALL_PATH --enable-shared --enable-optimized
make -j $MAKEJ
make install

cd ../..
