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
#    linux-armhf)
#        export CC_FLAGS="clang -target arm -march=armv7 -mfloat-abi=hard"
#        export CXX_FLAGS="-target arm -march=armv7 -mfloat-abi=hard"
#        ;;
    macosx-*)
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

LLVM_VERSION=3.9.0
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

cmake -DCMAKE_INSTALL_PREFIX=../.. -DDLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host ..
make -j $MAKEJ
make install

cd ../..
