#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" llvm
    popd
    exit
fi

LLVM_VERSION=10.0.1
download https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/llvm-$LLVM_VERSION.src.tar.xz llvm-$LLVM_VERSION.src.tar.xz
download https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/clang-$LLVM_VERSION.src.tar.xz clang-$LLVM_VERSION.src.tar.xz
download https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/polly-$LLVM_VERSION.src.tar.xz polly-$LLVM_VERSION.src.tar.xz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives... (ignore any symlink errors)"
tar --totals -xf ../llvm-$LLVM_VERSION.src.tar.xz || tar --totals -xf ../llvm-$LLVM_VERSION.src.tar.xz
cd llvm-$LLVM_VERSION.src
mkdir -p build tools
cd tools
tar --totals -xf ../../../clang-$LLVM_VERSION.src.tar.xz || tar --totals -xf ../../../clang-$LLVM_VERSION.src.tar.xz
rm -Rf clang
mv clang-$LLVM_VERSION.src clang
tar --totals -xf ../../../polly-$LLVM_VERSION.src.tar.xz || tar --totals -xf ../../../polly-$LLVM_VERSION.src.tar.xz
rm -Rf polly
mv polly-$LLVM_VERSION.src polly
sedinplace '/Generating libLLVM is not supported on MSVC/d' llvm-shlib/CMakeLists.txt
cd ../build

case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    linux-armhf)
        mkdir -p ../tblgen
        cd ../tblgen
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_DEFAULT_TARGET_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_TARGET_ARCH=ARM -DLLVM_TARGETS_TO_BUILD=ARM -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        cd ../build
        CC="arm-linux-gnueabihf-gcc" CXX="arm-linux-gnueabihf-g++" $CMAKE -DCMAKE_EXE_LINKER_FLAGS="-ldl" -DCMAKE_SHARED_LINKER_FLAGS="-ldl" -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_DEFAULT_TARGET_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_TARGET_ARCH=ARM -DLLVM_TARGETS_TO_BUILD=ARM -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    linux-arm64)
        mkdir -p ../tblgen
        cd ../tblgen
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_TARGET_ARCH=AArch64 -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        cd ../build
        CC="aarch64-linux-gnu-gcc" CXX="aarch64-linux-gnu-g++" $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_TARGET_ARCH=AArch64 -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    linux-ppc64le)
        mkdir -p ../tblgen
        cd ../tblgen
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_TARGET_ARCH=PowerPC -DLLVM_TARGETS_TO_BUILD=PowerPC -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        cd ../build
        CC="powerpc64le-linux-gnu-gcc" CXX="powerpc64le-linux-gnu-g++" $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_TARGET_ARCH=PowerPC -DLLVM_TARGETS_TO_BUILD=PowerPC -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    macosx-*)
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,@loader_path/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF ..
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    windows-x86)
        $CMAKE -G "Visual Studio 15 2017" -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="C:/Python27/python.exe" -DLLVM_POLLY_LINK_INTO_TOOLS=ON ..
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        cd Release/lib/
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        cd ../..
        $CMAKE -G "Visual Studio 15 2017" -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="C:/Python27/python.exe" -DLLVM_POLLY_LINK_INTO_TOOLS=ON ..
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        cd ../../lib
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        [ -f LTO.lib ] || cp ../llvm-$LLVM_VERSION.src/build/Release/lib/LTO.lib .
        cd ../llvm-$LLVM_VERSION.src/build
        ;;
    windows-x86_64)
        $CMAKE -G "Visual Studio 15 2017 Win64" -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -Thost=x64 -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="C:/Python27/python.exe" -DLLVM_POLLY_LINK_INTO_TOOLS=ON ..
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        cd Release/lib/
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        cd ../..
        $CMAKE -G "Visual Studio 15 2017 Win64" -Thost=x64 -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="C:/Python27/python.exe" -DLLVM_POLLY_LINK_INTO_TOOLS=ON ..
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
