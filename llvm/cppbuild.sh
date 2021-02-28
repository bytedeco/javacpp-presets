#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" llvm
    popd
    exit
fi

LLVM_VERSION=11.1.0
download https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/llvm-project-$LLVM_VERSION.src.tar.xz llvm-project-$LLVM_VERSION.src.tar.xz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives... (ignore any symlink errors)"
tar --totals -xf ../llvm-project-$LLVM_VERSION.src.tar.xz || true
cd llvm-project-$LLVM_VERSION.src
patch -Np1 < ../../../llvm.patch
sedinplace '/find_package(Git/d' llvm/cmake/modules/AddLLVM.cmake llvm/cmake/modules/VersionFromVCS.cmake
sedinplace '/Generating libLLVM is not supported on MSVC/d' llvm/tools/llvm-shlib/CMakeLists.txt
mkdir -p build
cd build

PROJECTS="clang;lld;polly"

case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    linux-armhf)
        mkdir -p ../tblgen
        cd ../tblgen
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_DEFAULT_TARGET_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_TARGET_ARCH=ARM -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        make -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        cd ../build
        CC="arm-linux-gnueabihf-gcc" CXX="arm-linux-gnueabihf-g++" $CMAKE -DCMAKE_EXE_LINKER_FLAGS="-ldl" -DCMAKE_SHARED_LINKER_FLAGS="-ldl" -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_DEFAULT_TARGET_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_TARGET_ARCH=ARM -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    linux-arm64)
        mkdir -p ../tblgen
        cd ../tblgen
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_TARGET_ARCH=AArch64 -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        make -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        cd ../build
        CC="aarch64-linux-gnu-gcc" CXX="aarch64-linux-gnu-g++" $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_TARGET_ARCH=AArch64 -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    linux-ppc64le)
        mkdir -p ../tblgen
        cd ../tblgen
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_TARGET_ARCH=PowerPC -DLLVM_TARGETS_TO_BUILD=PowerPC -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        make -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        cd ../build
        CC="powerpc64le-linux-gnu-gcc" CXX="powerpc64le-linux-gnu-g++" $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_TARGET_ARCH=PowerPC -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    macosx-*)
        $CMAKE -DLLVM_CCACHE_BUILD=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,@loader_path/' -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        make -j $MAKEJ
        make install > /dev/null
        cp lib/LLVMPolly.so ../../lib/libLLVMPolly.so
        ;;
    windows-x86)
        export CC="cl.exe"
        export CXX="cl.exe"
        $CMAKE -G "Ninja" -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$(where python.exe | head -1)" -DLLVM_POLLY_LINK_INTO_TOOLS=ON -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        ninja -j $MAKEJ
        cd lib/
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib Polly*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        cd ..
        $CMAKE -G "Ninja" -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$(where python.exe | head -1)" -DLLVM_POLLY_LINK_INTO_TOOLS=ON -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        ninja -j $MAKEJ
        ninja install
        cd ../../lib
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib Polly*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        [ -f LTO.lib ] || cp ../llvm-$LLVM_VERSION.src/build/lib/LTO.lib .
        cd ../llvm-project-$LLVM_VERSION.src/build
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"
        $CMAKE -G "Ninja" -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$(where python.exe | head -1)" -DLLVM_POLLY_LINK_INTO_TOOLS=ON -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        ninja -j $MAKEJ
        cd lib/
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib Polly*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        cd ..
        $CMAKE -G "Ninja" -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=../.. -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$(where python.exe | head -1)" -DLLVM_POLLY_LINK_INTO_TOOLS=ON -DLLVM_ENABLE_PROJECTS="$PROJECTS" ../llvm
        ninja -j $MAKEJ
        ninja install
        cd ../../lib
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib Polly*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        [ -f LTO.lib ] || cp ../llvm-$LLVM_VERSION.src/build/lib/LTO.lib .
        cd ../llvm-project-$LLVM_VERSION.src/build
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

cd ../..
