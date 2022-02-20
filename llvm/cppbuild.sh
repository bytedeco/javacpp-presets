#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" llvm
    popd
    exit
fi

LLVM_VERSION=13.0.1
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
sedinplace 's/if (NOT Python3_EXECUTABLE/if (TRUE/g' clang/lib/Tooling/CMakeLists.txt
mkdir -p build
cd build

PROJECTS="clang;lld;polly"

TBLGEN_BUILD="${BUILD_DIR:-$(pwd)}/../tblgen"
LLVM_BUILD="${BUILD_DIR:-$(pwd)}"

case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        $CMAKE -S ../llvm -B $LLVM_BUILD -DLLVM_CCACHE_BUILD=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        make -C $LLVM_BUILD -j $MAKEJ
        make -C $LLVM_BUILD install > /dev/null
        cp $INSTALL_PATH/lib/LLVMPolly.so $INSTALL_PATH/lib/libLLVMPolly.so
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        $CMAKE -S ../llvm -B $LLVM_BUILD -DLLVM_CCACHE_BUILD=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        make -C $LLVM_BUILD -j $MAKEJ
        make -C $LLVM_BUILD install > /dev/null
        cp $INSTALL_PATH/lib/LLVMPolly.so $INSTALL_PATH/lib/libLLVMPolly.so
        ;;
    linux-armhf)
        mkdir -p ../tblgen
        cd ../tblgen
        $CMAKE -S ../llvm -B $TBLGEN_BUILD -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_DEFAULT_TARGET_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_TARGET_ARCH=ARM -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        make -C $TBLGEN_BUILD -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        cd ../build
        export CC="arm-linux-gnueabihf-gcc"
        export CXX="arm-linux-gnueabihf-g++"
        $CMAKE -S ../llvm -B $LLVM_BUILD -DCMAKE_EXE_LINKER_FLAGS="-ldl" -DCMAKE_SHARED_LINKER_FLAGS="-ldl" -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_DEFAULT_TARGET_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_TARGET_ARCH=ARM -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        make -C $LLVM_BUILD -j $MAKEJ
        make -C $LLVM_BUILD install > /dev/null
        cp $INSTALL_PATH/lib/LLVMPolly.so $INSTALL_PATH/lib/libLLVMPolly.so
        ;;
    linux-arm64)
        mkdir -p ../tblgen
        cd ../tblgen
        $CMAKE -S ../llvm -B $TBLGEN_BUILD -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_TARGET_ARCH=AArch64 -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        make -C $TBLGEN_BUILD -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        cd ../build
        export CC="aarch64-linux-gnu-gcc"
        export CXX="aarch64-linux-gnu-g++"
        $CMAKE -S ../llvm -B $LLVM_BUILD -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_TARGET_ARCH=AArch64 -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        make -C $LLVM_BUILD -j $MAKEJ
        make -C $LLVM_BUILD install > /dev/null
        cp $INSTALL_PATH/lib/LLVMPolly.so $INSTALL_PATH/lib/libLLVMPolly.so
        ;;
    linux-ppc64le)
        mkdir -p ../tblgen
        cd ../tblgen
        $CMAKE -S ../llvm -B $TBLGEN_BUILD -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_TARGET_ARCH=PowerPC -DLLVM_TARGETS_TO_BUILD=PowerPC -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        make -C $TBLGEN_BUILD -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        cd ../build
        export CC="powerpc64le-linux-gnu-gcc"
        export CXX="powerpc64le-linux-gnu-g++"
        $CMAKE -S ../llvm -B $LLVM_BUILD -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath,$ORIGIN/' -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_TARGET_ARCH=PowerPC -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        make -C $LLVM_BUILD -j $MAKEJ
        make -C $LLVM_BUILD install > /dev/null
        cp $INSTALL_PATH/lib/LLVMPolly.so $INSTALL_PATH/lib/libLLVMPolly.so
        ;;
    macosx-arm64)
        mkdir -p ../tblgen
        cd ../tblgen
        $CMAKE -S ../llvm -B $TBLGEN_BUILD -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=arm64-apple-darwin20.6.0 -DLLVM_DEFAULT_TARGET_TRIPLE=arm64-apple-darwin20.6.0 -DLLVM_TARGET_ARCH=AArch64 -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        make -C $TBLGEN_BUILD -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        cd ../build
        $CMAKE -S ../llvm -B $LLVM_BUILD -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DCMAKE_CXX_FLAGS='-arch arm64' -DCMAKE_C_FLAGS='-arch arm64' -DCMAKE_EXE_LINKER_FLAGS='-arch arm64 -Wl,-rpath,@loader_path/' -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=arm64-apple-darwin20.6.0 -DLLVM_DEFAULT_TARGET_TRIPLE=arm64-apple-darwin20.6.0 -DLLVM_TARGET_ARCH=AArch64 -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        make -C $LLVM_BUILD -j $MAKEJ
        make -C $LLVM_BUILD install > /dev/null
        cp $INSTALL_PATH/lib/LLVMPolly.so $INSTALL_PATH/lib/libLLVMPolly.so
        ;;
    macosx-x86_64)
        $CMAKE -S ../llvm -B $LLVM_BUILD -DLLVM_CCACHE_BUILD=ON -DCMAKE_CXX_FLAGS='-arch x86_64' -DCMAKE_C_FLAGS='-arch x86_64' -DCMAKE_EXE_LINKER_FLAGS='-arch x86_64 -Wl,-rpath,@loader_path/' -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        make -C $LLVM_BUILD -j $MAKEJ
        make -C $LLVM_BUILD install > /dev/null
        cp $INSTALL_PATH/lib/LLVMPolly.so $INSTALL_PATH/lib/libLLVMPolly.so
        ;;
    windows-x86)
        export INSTALL_PATH=$(cygpath -w $INSTALL_PATH)
        export CC="cl.exe"
        export CXX="cl.exe"
        $CMAKE -G "Ninja" -S ../llvm -B $LLVM_BUILD -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$(where python.exe | head -1)" -DLLVM_POLLY_LINK_INTO_TOOLS=ON -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        ninja -C $LLVM_BUILD -j $MAKEJ
        pushd $LLVM_BUILD/lib
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib Polly*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        popd
        $CMAKE -G "Ninja" -S ../llvm -B $LLVM_BUILD -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$(where python.exe | head -1)" -DLLVM_POLLY_LINK_INTO_TOOLS=ON -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        ninja -C $LLVM_BUILD -j $MAKEJ
        ninja -C $LLVM_BUILD install
        pushd $INSTALL_PATH/lib
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib Polly*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        [ -f LTO.lib ] || cp ../llvm-$LLVM_VERSION.src/build/lib/LTO.lib .
        popd
        ;;
    windows-x86_64)
        export INSTALL_PATH=$(cygpath -w $INSTALL_PATH)
        export CC="cl.exe"
        export CXX="cl.exe"
        $CMAKE -G "Ninja" -S ../llvm -B $LLVM_BUILD -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$(where python.exe | head -1)" -DLLVM_POLLY_LINK_INTO_TOOLS=ON -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        ninja -C $LLVM_BUILD -j $MAKEJ
        pushd $LLVM_BUILD/lib
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib Polly*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        popd
        $CMAKE -G "Ninja" -S ../llvm -B $LLVM_BUILD -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DLLVM_BUILD_LLVM_C_DYLIB=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$(where python.exe | head -1)" -DLLVM_POLLY_LINK_INTO_TOOLS=ON -DLLVM_ENABLE_PROJECTS="$PROJECTS"
        ninja -C $LLVM_BUILD -j $MAKEJ
        ninja -C $LLVM_BUILD install
        pushd $INSTALL_PATH/lib
        [ -f LLVM.lib ] || lib.exe /OUT:LLVM.lib LLVM*.lib Polly*.lib
        [ -f clang.lib ] || lib.exe /OUT:clang.lib clang*.lib
        [ -f LTO.lib ] || cp ../llvm-$LLVM_VERSION.src/build/lib/LTO.lib .
        popd
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

cd ../..
