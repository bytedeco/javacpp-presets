#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" arrow
    popd
    exit
fi

export PYTHON_BIN_PATH=$(which python3)
if [[ $PLATFORM == windows* ]]; then
    export PYTHON_BIN_PATH=$(which python.exe)
fi

LLVM_VERSION=15.0.3
OPENSSL_VERSION=3.0.5
ZLIB_VERSION=1.2.13
PROTO_VERSION=3.17.3 # cpp/thirdparty/versions.txt
ARROW_VERSION=6.0.1
download https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/llvm-project-$LLVM_VERSION.src.tar.xz llvm-project-$LLVM_VERSION.src.tar.xz
download https://github.com/python/cpython-bin-deps/archive/openssl-bin.zip cpython-bin-deps-openssl-bin.zip
download https://www.openssl.org/source/openssl-$OPENSSL_VERSION.tar.gz openssl-$OPENSSL_VERSION.tar.gz
download http://zlib.net/fossils/zlib-$ZLIB_VERSION.tar.gz zlib-$ZLIB_VERSION.tar.gz
download https://github.com/google/protobuf/releases/download/v$PROTO_VERSION/protobuf-cpp-$PROTO_VERSION.tar.gz protobuf-$PROTO_VERSION.tar.gz
download https://archive.apache.org/dist/arrow/arrow-$ARROW_VERSION/apache-arrow-$ARROW_VERSION.tar.gz apache-arrow-$ARROW_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives... (ignore any symlink errors)"
tar --totals -xzf ../apache-arrow-$ARROW_VERSION.tar.gz
tar --totals -xzf ../protobuf-$PROTO_VERSION.tar.gz
tar --totals -xzf ../zlib-$ZLIB_VERSION.tar.gz
tar --totals -xzf ../openssl-$OPENSSL_VERSION.tar.gz
tar --totals -xf ../llvm-project-$LLVM_VERSION.src.tar.xz || true
cd apache-arrow-$ARROW_VERSION
patch -Np1 < ../../../arrow.patch
sedinplace 's/PlatformToolset=v140/PlatformToolset=v142/g' cpp/cmake_modules/ThirdpartyToolchain.cmake
sedinplace 's/set(ARROW_LLVM_VERSIONS/set(ARROW_LLVM_VERSIONS "15.0"/g' cpp/CMakeLists.txt
sedinplace 's:llvm/Support/TargetRegistry.h:llvm/MC/TargetRegistry.h:g' cpp/src/gandiva/engine.cc
sedinplace 's/message(FATAL_ERROR "Unsupported MSVC_VERSION=${MSVC_VERSION}")/set(FMS_COMPATIBILITY 19.20)/g' cpp/src/gandiva/precompiled/CMakeLists.txt
cd ../llvm-project-$LLVM_VERSION.src
sedinplace '/find_package(Git/d' llvm/cmake/modules/AddLLVM.cmake llvm/cmake/modules/VersionFromVCS.cmake
mkdir -p build
cd build

COMPONENTS="-DARROW_COMPUTE=ON -DARROW_CSV=ON -DARROW_DATASET=ON -DARROW_FILESYSTEM=ON -DARROW_FLIGHT=ON -DARROW_GANDIVA=ON -DARROW_HDFS=ON -DARROW_JSON=ON -DARROW_ORC=ON -DARROW_PARQUET=ON -DARROW_PLASMA=ON -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_VERBOSE_THIRDPARTY_BUILD=ON -DARROW_USE_GLOG=ON -DARROW_WITH_BROTLI=ON -DARROW_WITH_BZ2=OFF -DARROW_WITH_LZ4=ON -DARROW_WITH_SNAPPY=ON -DARROW_WITH_ZLIB=ON -DARROW_WITH_ZSTD=OFF"

case $PLATFORM in
    linux-armhf)
        mkdir -p ../tblgen
        cd ../tblgen
        "$CMAKE" -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_DEFAULT_TARGET_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_TARGET_ARCH=ARM -DLLVM_TARGETS_TO_BUILD=ARM -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_PROJECTS="clang" ../llvm
        make -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        mkdir -p ../build
        cd ../build
        export CC="arm-linux-gnueabihf-gcc"
        export CXX="arm-linux-gnueabihf-g++ -std=c++11"
        "$CMAKE" -DCMAKE_EXE_LINKER_FLAGS="-ldl" -DCMAKE_SHARED_LINKER_FLAGS="-ldl" -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_DEFAULT_TARGET_TRIPLE=arm-unknown-linux-gnueabihf -DLLVM_TARGET_ARCH=ARM -DLLVM_TARGETS_TO_BUILD=ARM -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_PROJECTS="clang" ../llvm
        make -j $MAKEJ
        make install
        cd ../../openssl-$OPENSSL_VERSION
        ./Configure linux-generic32 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib --cross-compile-prefix=arm-linux-gnueabihf-
        make -s -j $MAKEJ
        make install_sw
        cd ../apache-arrow-$ARROW_VERSION/cpp
        "$CMAKE" -DCMAKE_C_FLAGS="-std=c99" -DCMAKE_CXX_FLAGS="-std=c++11" -DLLVM_ROOT="$INSTALL_PATH" -DOPENSSL_ROOT_DIR="$INSTALL_PATH" $COMPONENTS -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF -DPython3_EXECUTABLE="$PYTHON_BIN_PATH" .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-arm64)
        mkdir -p ../tblgen
        cd ../tblgen
        "$CMAKE" -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_TARGET_ARCH=AArch64 -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_PROJECTS="clang" ../llvm
        make -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        mkdir -p ../build
        cd ../build
        export CC="aarch64-linux-gnu-gcc"
        export CXX="aarch64-linux-gnu-g++ -std=c++11"
        "$CMAKE" -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-unknown-linux-gnu -DLLVM_TARGET_ARCH=AArch64 -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_PROJECTS="clang" ../llvm
        make -j $MAKEJ
        make install
        cd ../../openssl-$OPENSSL_VERSION
        ./Configure linux-aarch64 -march=armv8-a+crypto -mcpu=cortex-a57+crypto -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib --cross-compile-prefix=aarch64-linux-gnu-
        make -s -j $MAKEJ
        make install_sw
        cd ../apache-arrow-$ARROW_VERSION/cpp
        "$CMAKE" -DCMAKE_C_FLAGS="-mabi=lp64" -DCMAKE_CXX_FLAGS="-std=c++11 -mabi=lp64" -DLLVM_ROOT="$INSTALL_PATH" -DOPENSSL_ROOT_DIR="$INSTALL_PATH" $COMPONENTS -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF -DPython3_EXECUTABLE="$PYTHON_BIN_PATH" .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-ppc64le)
        mkdir -p ../tblgen
        cd ../tblgen
        "$CMAKE" -DLLVM_CCACHE_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_TARGET_ARCH=PowerPC -DLLVM_TARGETS_TO_BUILD=PowerPC -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_PROJECTS="clang" ../llvm
        make -j $MAKEJ llvm-tblgen clang-tblgen
        TBLGEN=`pwd`
        mkdir -p ../build
        cd ../build
        export CC="powerpc64le-linux-gnu-gcc"
        export CXX="powerpc64le-linux-gnu-g++ -std=c++11"
        "$CMAKE" -DLLVM_CCACHE_BUILD=ON -DCMAKE_CROSSCOMPILING=True -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DLLVM_TABLEGEN="$TBLGEN/bin/llvm-tblgen" -DCLANG_TABLEGEN="$TBLGEN/bin/clang-tblgen" -DCMAKE_BUILD_TYPE=Release -DLLVM_HOST_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_DEFAULT_TARGET_TRIPLE=powerpc64le-unknown-linux-gnu -DLLVM_TARGET_ARCH=PowerPC -DLLVM_TARGETS_TO_BUILD=PowerPC -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_PROJECTS="clang" ../llvm
        make -j $MAKEJ
        make install
        cd ../../openssl-$OPENSSL_VERSION
        ./Configure linux-ppc64le -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib --cross-compile-prefix=powerpc64le-linux-gnu-
        make -s -j $MAKEJ
        make install_sw
        cd ../apache-arrow-$ARROW_VERSION/cpp
        "$CMAKE" -DCMAKE_C_FLAGS="-m64" -DCMAKE_CXX_FLAGS="-std=c++11 -m64" -DLLVM_ROOT="$INSTALL_PATH" -DOPENSSL_ROOT_DIR="$INSTALL_PATH" $COMPONENTS -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF -DPython3_EXECUTABLE="$PYTHON_BIN_PATH" .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -std=c++11 -m32"
        "$CMAKE" -DLLVM_CCACHE_BUILD=ON -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_PROJECTS="clang" ../llvm
        make -j $MAKEJ
        make install
        cd ../../openssl-$OPENSSL_VERSION
        ./Configure linux-elf -m32 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../apache-arrow-$ARROW_VERSION/cpp
        "$CMAKE" -DCMAKE_C_FLAGS="-m32" -DCMAKE_CXX_FLAGS="-std=c++11 -m32" -DLLVM_ROOT="$INSTALL_PATH" -DOPENSSL_ROOT_DIR="$INSTALL_PATH" $COMPONENTS -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF -DPython3_EXECUTABLE="$PYTHON_BIN_PATH" .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -std=c++11 -m64"
        "$CMAKE" -DLLVM_CCACHE_BUILD=ON -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_PROJECTS="clang" ../llvm
        make -j $MAKEJ
        make install
        cd ../../openssl-$OPENSSL_VERSION
        ./Configure linux-x86_64 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../apache-arrow-$ARROW_VERSION/cpp
        "$CMAKE" -DCMAKE_C_FLAGS="-m64" -DCMAKE_CXX_FLAGS="-std=c++11 -m64" -DLLVM_ROOT="$INSTALL_PATH" -DOPENSSL_ROOT_DIR="$INSTALL_PATH" $COMPONENTS -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF -DPython3_EXECUTABLE="$PYTHON_BIN_PATH" .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-*)
        export CC="clang"
        export CXX="clang++"
        "$CMAKE" -DLLVM_CCACHE_BUILD=ON -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_PROJECTS="clang" ../llvm
        make -j $MAKEJ
        make install
        cd ../../openssl-$OPENSSL_VERSION
        ./Configure darwin64-x86_64-cc -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../apache-arrow-$ARROW_VERSION/cpp
        sedinplace 's/"cxxflags=-fPIC"/"toolset=clang" "cxxflags=-fPIC"/g' cmake_modules/ThirdpartyToolchain.cmake
        "$CMAKE" -DCLANG_EXECUTABLE="/usr/bin/clang" -DLLVM_ROOT="$INSTALL_PATH" -DOPENSSL_ROOT_DIR="$INSTALL_PATH" $COMPONENTS -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF -DPython3_EXECUTABLE="$PYTHON_BIN_PATH" .
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86)
        export CC="cl.exe"
        export CXX="cl.exe"
        "$CMAKE" -G "Ninja" -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_PROJECTS="clang" ../llvm
        ninja -j $MAKEJ
        ninja install
        cd ../..
        unzip -o ../cpython-bin-deps-openssl-bin.zip
        cd cpython-bin-deps-openssl-bin/
        cp -r win32/include/* ../include
        cp win32/*.dll ../bin
        cp win32/*.lib ../lib
        cd ../zlib-$ZLIB_VERSION
        nmake -f win32/Makefile.msc zlib.lib
        cp *.h ../include
        cp zlib.lib ../lib
        cd ../protobuf-$PROTO_VERSION
        "$CMAKE" -G "Ninja" -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DZLIB_INCLUDE_DIR="$INSTALL_PATH/include" -DZLIB_LIB="$INSTALL_PATH/lib" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" cmake
        ninja -j $MAKEJ
        ninja install
        cd ..
        sedinplace 's/PROTOBUF_USE_DLLS/DISABLE_PROTOBUF_USE_DLLS/g' include/google/protobuf/*.h include/google/protobuf/*.inc include/google/protobuf/stubs/*
        sedinplace "s:Ws2_32.lib:../../../lib/zlib Ws2_32.lib:g" apache-arrow-$ARROW_VERSION/cpp/src/arrow/flight/CMakeLists.txt
        "$CMAKE" -G "Ninja" -DLLVM_ROOT="$INSTALL_PATH" -DOPENSSL_ROOT_DIR="$INSTALL_PATH" -DARROW_PROTOBUF_USE_SHARED=OFF -DProtobuf_SOURCE=SYSTEM -DZLIB_SOURCE=SYSTEM $COMPONENTS -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF -DPython3_EXECUTABLE="$PYTHON_BIN_PATH" apache-arrow-$ARROW_VERSION/cpp
        ninja -j $MAKEJ
        ninja install
        cd apache-arrow-$ARROW_VERSION/cpp
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"
        "$CMAKE" -G "Ninja" -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON -DLLVM_USE_CRT_RELEASE=MD -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_DIA_SDK=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_PROJECTS="clang" ../llvm
        ninja -j $MAKEJ
        ninja install
        cd ../..
        unzip -o ../cpython-bin-deps-openssl-bin.zip
        cd cpython-bin-deps-openssl-bin/
        cp -r amd64/include/* ../include
        cp amd64/*.dll ../bin
        cp amd64/*.lib ../lib
        cd ../zlib-$ZLIB_VERSION
        nmake -f win32/Makefile.msc zlib.lib
        cp *.h ../include
        cp zlib.lib ../lib
        cd ../protobuf-$PROTO_VERSION
        "$CMAKE" -G "Ninja" -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DZLIB_INCLUDE_DIR="$INSTALL_PATH/include" -DZLIB_LIB="$INSTALL_PATH/lib" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF -DPYTHON_EXECUTABLE="$PYTHON_BIN_PATH" cmake
        ninja -j $MAKEJ
        ninja install
        cd ..
        sedinplace 's/PROTOBUF_USE_DLLS/DISABLE_PROTOBUF_USE_DLLS/g' include/google/protobuf/*.h include/google/protobuf/*.inc include/google/protobuf/stubs/*
        sedinplace "s:Ws2_32.lib:../../../lib/zlib Ws2_32.lib:g" apache-arrow-$ARROW_VERSION/cpp/src/arrow/flight/CMakeLists.txt
        "$CMAKE" -G "Ninja" -DLLVM_ROOT="$INSTALL_PATH" -DOPENSSL_ROOT_DIR="$INSTALL_PATH" -DARROW_PROTOBUF_USE_SHARED=OFF -DProtobuf_SOURCE=SYSTEM -DZLIB_SOURCE=SYSTEM $COMPONENTS -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF -DPython3_EXECUTABLE="$PYTHON_BIN_PATH" apache-arrow-$ARROW_VERSION/cpp
        ninja -j $MAKEJ
        ninja install
        cd apache-arrow-$ARROW_VERSION/cpp
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

# work around link issues on Windows
echo "extern template class arrow::Future<std::shared_ptr<arrow::RecordBatch> >;"  >> ../../include/arrow/dataset/scanner.h
echo "extern template class arrow::Future<arrow::dataset::TaggedRecordBatch>;"     >> ../../include/arrow/dataset/scanner.h
echo "extern template class arrow::Future<arrow::dataset::EnumeratedRecordBatch>;" >> ../../include/arrow/dataset/scanner.h

cd ../../..
