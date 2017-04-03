#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" llvm
    popd
    exit
fi

LLVM_VERSION=4.0.0
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

case $PLATFORM in
    windows-x86)
        "$CMAKE" -G "Visual Studio 14 2015" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLLVM_USE_CRT_RELEASE="MD" -DCMAKE_BUILD_TYPE="Release" -DLLVM_TARGETS_TO_BUILD="host" ..
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //m

        cp Release/lib/LTO.lib $INSTALL_PATH/lib

        cd ../../lib

        lib.exe /OUT:LLVM.lib LLVM*.lib
        lib.exe /OUT:clang.lib clang*.lib

        ;;
    windows-x86_64)
        "$CMAKE" -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLLVM_USE_CRT_RELEASE="MD" -DCMAKE_BUILD_TYPE="Release" -DLLVM_TARGETS_TO_BUILD="host" ..
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //m

        cp Release/lib/LTO.lib $INSTALL_PATH/lib

        cd ../../lib

        lib.exe /OUT:LLVM.lib LLVM*.lib
        lib.exe /OUT:clang.lib clang*.lib

        ;;
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"

        $CMAKE -DCMAKE_INSTALL_PREFIX=../.. -DDLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLIBXML2_LIBRARIES= ..
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"

        $CMAKE -DCMAKE_INSTALL_PREFIX=../.. -DDLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLIBXML2_LIBRARIES= ..
        make -j $MAKEJ
        make install
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

cd ../..