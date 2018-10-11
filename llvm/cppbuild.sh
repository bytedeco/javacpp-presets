#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" llvm
    popd
    exit
fi

LLVM_VERSION=7.0.0
download http://llvm.org/releases/$LLVM_VERSION/llvm-$LLVM_VERSION.src.tar.xz llvm-$LLVM_VERSION.src.tar.xz
download http://llvm.org/releases/$LLVM_VERSION/cfe-$LLVM_VERSION.src.tar.xz cfe-$LLVM_VERSION.src.tar.xz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives... (ignore any symlink errors)"
tar --totals -xf ../llvm-$LLVM_VERSION.src.tar.xz || tar --totals -xf ../llvm-$LLVM_VERSION.src.tar.xz
cd llvm-$LLVM_VERSION.src
mkdir -p build tools
cd tools
tar --totals -xf ../../../cfe-$LLVM_VERSION.src.tar.xz || tar --totals -xf ../../../cfe-$LLVM_VERSION.src.tar.xz
rm -Rf clang
mv cfe-$LLVM_VERSION.src clang
cd ../build

case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        $CMAKE -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLIBXML2_LIBRARIES= -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ
        make install > /dev/null
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        $CMAKE -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLIBXML2_LIBRARIES= -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ
        make install > /dev/null
        ;;
#    linux-armhf)
#        export CC_FLAGS="clang -target arm -march=armv7 -mfloat-abi=hard"
#        export CXX_FLAGS="-target arm -march=armv7 -mfloat-abi=hard"
#        $CMAKE -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLIBXML2_LIBRARIES= -DLLVM_INCLUDE_TESTS=OFF ..
#        make -j $MAKEJ
#        make install > /dev/null
#        ;;
    macosx-*)
        $CMAKE -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLIBXML2_LIBRARIES= -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ
        make install > /dev/null
        ;;
    windows-x86)
        $CMAKE -G "Visual Studio 14 2015" -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_DIA_SDK=OFF -DLIBXML2_LIBRARIES= -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="C:/Python27/python.exe" ..
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        cd Release/lib/
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        cd ../..
        $CMAKE -G "Visual Studio 14 2015" -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_DIA_SDK=OFF -DLIBXML2_LIBRARIES= -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="C:/Python27/python.exe" ..
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        cd ../../lib
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        [ -f LTO.lib ] || cp ../llvm-$LLVM_VERSION.src/build/Release/lib/LTO.lib .
        cd ../llvm-$LLVM_VERSION.src/build
        ;;
    windows-x86_64)
        $CMAKE -G "Visual Studio 14 2015 Win64" -Thost=x64 -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_DIA_SDK=OFF -DLIBXML2_LIBRARIES= -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="C:/Python27/python.exe" ..
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        cd Release/lib/
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        cd ../..
        $CMAKE -G "Visual Studio 14 2015 Win64" -Thost=x64 -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_DIA_SDK=OFF -DLIBXML2_LIBRARIES= -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="C:/Python27/python.exe" ..
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        cd ../../lib
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        [ -f LTO.lib ] || cp ../llvm-$LLVM_VERSION.src/build/Release/lib/LTO.lib .
        cd ../llvm-$LLVM_VERSION.src/build
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

cd ../..
