#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" llvm
    popd
    exit
fi

LLVM_VERSION=3.4.2
download http://llvm.org/releases/$LLVM_VERSION/llvm-$LLVM_VERSION.src.tar.gz llvm-$LLVM_VERSION.src.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
tar -xzvf ../llvm-$LLVM_VERSION.src.tar.gz
cd llvm-$LLVM_VERSION.src

case $PLATFORM in
    linux-x86)
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-optimized CC="clang -m32" CXX="clang++ -m32"
        make -j$NCPUS
        make install
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-optimized CC="clang -m64" CXX="clang++ -m64"
        make -j$NCPUS
        make install
        ;;
    macosx-*)
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-optimized
        make -j$NCPUS
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
