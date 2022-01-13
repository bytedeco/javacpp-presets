#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" dnnl
    popd
    exit
fi

export DNNL_CPU_RUNTIME="OMP" # or TBB
export DNNL_GPU_RUNTIME="OCL"

TBB_VERSION=2020.3
MKLDNN_VERSION=2.5.2
download https://github.com/oneapi-src/oneTBB/archive/v$TBB_VERSION.tar.gz oneTBB-$TBB_VERSION.tar.bz2
download https://github.com/oneapi-src/oneDNN/archive/v$MKLDNN_VERSION.tar.gz oneDNN-$MKLDNN_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib bin
INSTALL_PATH=`pwd`

OPENCL_PATH="$INSTALL_PATH/../../../opencl/cppbuild/$PLATFORM/"

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -f "$P/include/CL/cl.h" ]]; then
            OPENCL_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

OPENCL_PATH="${OPENCL_PATH//\\//}"

echo "Decompressing archives..."
tar --totals -xf ../oneTBB-$TBB_VERSION.tar.bz2
tar --totals -xf ../oneDNN-$MKLDNN_VERSION.tar.bz2
cd oneDNN-$MKLDNN_VERSION
patch -Np1 < ../../../mkl-dnn.patch

sedinplace 's/-fvisibility=internal//g' cmake/platform.cmake
sedinplace 's/-fvisibility-inlines-hidden//g' cmake/platform.cmake
sedinplace 's:Headers/cl.h:CL/cl.h:g' cmake/FindOpenCL.cmake

if [[ -d "$OPENCL_PATH" ]]; then
    export OPENCLROOT="$OPENCL_PATH"
fi

case $PLATFORM in
    linux-arm64)
        export CC="aarch64-linux-gnu-gcc"
        export CXX="aarch64-linux-gnu-g++"
        if [[ "$DNNL_CPU_RUNTIME" == "TBB" ]]; then
            cd ../oneTBB-$TBB_VERSION
            make -j $MAKEJ tbb_os=linux
            sedinplace 's/release/debug/g' Makefile
            make -j $MAKEJ tbb_os=linux
            cp -a include/* ../include
            cp -a build/*release/libtbb.* ../lib
            cp -a build/*debug/libtbb_debug.* ../lib
            strip ../lib/libtbb.so.*
            cd ../oneDNN-$MKLDNN_VERSION
        fi
        sedinplace 's/constexpr GRF     getBase/GRF getBase/g' src/gpu/jit/ngen/ngen_core.hpp
        sedinplace 's/constexpr int32_t getDisp/int32_t getDisp/g' src/gpu/jit/ngen/ngen_core.hpp
        sedinplace '/immintrin.h/d' src/gpu/jit/ngen/ngen_utils.hpp
        "$CMAKE" -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=AARCH64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' -DARCH_OPT_FLAGS='-Wno-error' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_CPU_RUNTIME=$DNNL_CPU_RUNTIME -DTBBROOT=$INSTALL_PATH -DDNNL_GPU_RUNTIME=$DNNL_GPU_RUNTIME .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        if [[ "$DNNL_CPU_RUNTIME" == "TBB" ]]; then
            cd ../oneTBB-$TBB_VERSION
            make -j $MAKEJ tbb_os=linux
            sedinplace 's/release/debug/g' Makefile
            make -j $MAKEJ tbb_os=linux
            cp -a include/* ../include
            cp -a build/*release/libtbb.* ../lib
            cp -a build/*debug/libtbb_debug.* ../lib
            strip ../lib/libtbb.so.*
            cd ../oneDNN-$MKLDNN_VERSION
        fi
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' -DARCH_OPT_FLAGS='-Wno-error' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_CPU_RUNTIME=$DNNL_CPU_RUNTIME -DTBBROOT=$INSTALL_PATH -DDNNL_GPU_RUNTIME=$DNNL_GPU_RUNTIME .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        if [[ "$DNNL_CPU_RUNTIME" == "TBB" ]]; then
            cd ../oneTBB-$TBB_VERSION
            make -j $MAKEJ tbb_os=macos
            sedinplace 's/release/debug/g' Makefile
            make -j $MAKEJ tbb_os=macos
            cp -a include/* ../include
            cp -a build/*release/libtbb.* ../lib
            cp -a build/*debug/libtbb_debug.* ../lib
            cd ../oneDNN-$MKLDNN_VERSION
        else
            mkdir -p ../lib
            cp /usr/local/lib/libomp.dylib ../lib/libiomp5.dylib
            chmod +w ../lib/libiomp5.dylib
            install_name_tool -id @rpath/libiomp5.dylib ../lib/libiomp5.dylib
        fi
        sedinplace 's/__thread/thread_local/g' src/common/utils.hpp
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DOpenMP_C_FLAG="-Xclang -fopenmp -I/usr/local/include -L$INSTALL_PATH/lib -liomp5" -DOpenMP_CXX_FLAG="-Xclang -fopenmp -I/usr/local/include -L$INSTALL_PATH/lib -liomp5" -DARCH_OPT_FLAGS='' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_CPU_RUNTIME=$DNNL_CPU_RUNTIME -DTBBROOT=$INSTALL_PATH .
        make -j $MAKEJ
        make install/strip
        install_name_tool -change @rpath/libomp.dylib @rpath/libiomp5.dylib ../lib/libdnnl.dylib
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"
        if [[ "$DNNL_CPU_RUNTIME" == "TBB" ]]; then
            cd ../oneTBB-$TBB_VERSION
            patch -Np1 < ../../../tbb-windows.patch
            make -j $MAKEJ tbb_os=windows runtime=vc14
            sedinplace 's/release/debug/g' Makefile
            make -j $MAKEJ tbb_os=windows runtime=vc14
            mkdir -p ../lib/intel64/vc14/
            cp -a include/* ../include
            cp -a build/*release/tbb.dll ../lib/
            cp -a build/*release/tbb.lib ../lib/intel64/vc14/
            cp -a build/*debug/tbb_debug.lib ../lib/intel64/vc14/
            cd ../oneDNN-$MKLDNN_VERSION
        fi
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR="lib" -DARCH_OPT_FLAGS='' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_CPU_RUNTIME=$DNNL_CPU_RUNTIME -DTBBROOT=$INSTALL_PATH -DDNNL_GPU_RUNTIME=$DNNL_GPU_RUNTIME .
        ninja -j $MAKEJ
        ninja install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
