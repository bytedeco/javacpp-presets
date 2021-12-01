#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" cpu_features
    popd
    exit
fi

CPU_FEATURES_VERSION=0.6.0
download https://github.com/google/cpu_features/archive/v$CPU_FEATURES_VERSION.tar.gz cpu_features-$CPU_FEATURES_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../cpu_features-$CPU_FEATURES_VERSION.tar.gz
cd cpu_features-$CPU_FEATURES_VERSION
patch -Np1 < ../../../cpu_features.patch
patch -Np1 < ../../../cpu_features-android.patch || true

case $PLATFORM in
    android-arm)
        sedinplace 's/CpuId(uint32_t leaf_id);/CpuId(uint32_t leaf_id) { }/g' include/internal/cpuid_x86.h
        sedinplace 's/GetXCR0Eax(void);/GetXCR0Eax(void) { }/g' include/internal/cpuid_x86.h
        sedinplace '/ANDROID_CPU_ARM_FEATURE_VFPv2/d' ndk_compat/cpu-features.c
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        make -j $MAKEJ
        make install
        ;;
    android-arm64)
        sedinplace 's/CpuId(uint32_t leaf_id);/CpuId(uint32_t leaf_id) { }/g' include/internal/cpuid_x86.h
        sedinplace 's/GetXCR0Eax(void);/GetXCR0Eax(void) { }/g' include/internal/cpuid_x86.h
        sedinplace '/ANDROID_CPU_ARM_FEATURE_VFPv2/d' ndk_compat/cpu-features.c
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        make -j $MAKEJ
        make install
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        make -j $MAKEJ
        make install
        ;;
    android-x86_64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        make -j $MAKEJ
        make install
        ;;
    linux-x86)
        sedinplace 's/sys\/auxv.h/linux\/auxvec.h/g' src/hwcaps.c
        CC="gcc -m32 -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        sedinplace 's/sys\/auxv.h/linux\/auxvec.h/g' src/hwcaps.c
        CC="gcc -m64 -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        make -j $MAKEJ
        make install
        ;;
    linux-armhf)
        sedinplace 's/CpuId(uint32_t leaf_id);/CpuId(uint32_t leaf_id) { }/g' include/internal/cpuid_x86.h
        sedinplace 's/GetXCR0Eax(void);/GetXCR0Eax(void) { }/g' include/internal/cpuid_x86.h
        sedinplace 's/set(PROCESSOR_IS_ARM FALSE)/set(PROCESSOR_IS_ARM TRUE)/g' CMakeLists.txt
        sedinplace 's/set(PROCESSOR_IS_X86 TRUE)/set(PROCESSOR_IS_X86 FALSE)/g' CMakeLists.txt
        sedinplace 's/(HardwareCapabilities)//g' src/define_tables.h
        CC="arm-linux-gnueabihf-gcc -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        make -j $MAKEJ
        make install
        ;;
    linux-arm64)
        sedinplace 's/CpuId(uint32_t leaf_id);/CpuId(uint32_t leaf_id) { }/g' include/internal/cpuid_x86.h
        sedinplace 's/GetXCR0Eax(void);/GetXCR0Eax(void) { }/g' include/internal/cpuid_x86.h
        sedinplace 's/set(PROCESSOR_IS_AARCH64 FALSE)/set(PROCESSOR_IS_AARCH64 TRUE)/g' CMakeLists.txt
        sedinplace 's/set(PROCESSOR_IS_X86 TRUE)/set(PROCESSOR_IS_X86 FALSE)/g' CMakeLists.txt
        CC="aarch64-linux-gnu-gcc -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        make -j $MAKEJ
        make install
        ;;
    linux-ppc64le)
        sedinplace 's/CpuId(uint32_t leaf_id);/CpuId(uint32_t leaf_id) { }/g' include/internal/cpuid_x86.h
        sedinplace 's/GetXCR0Eax(void);/GetXCR0Eax(void) { }/g' include/internal/cpuid_x86.h
        sedinplace 's/set(PROCESSOR_IS_POWER FALSE)/set(PROCESSOR_IS_POWER TRUE)/g' CMakeLists.txt
        sedinplace 's/set(PROCESSOR_IS_X86 TRUE)/set(PROCESSOR_IS_X86 FALSE)/g' CMakeLists.txt
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64 -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        else
          CC="powerpc64le-linux-gnu-gcc -fPIC" CMAKE_C_COMPILER=$CC $CMAKE -DCMAKE_SYSTEM_PROCESSOR=powerpc -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        fi
        make -j $MAKEJ
        make install
        ;;
    linux-mips64el)
        sedinplace 's/CpuId(uint32_t leaf_id);/CpuId(uint32_t leaf_id) { }/g' include/internal/cpuid_x86.h
        sedinplace 's/GetXCR0Eax(void);/GetXCR0Eax(void) { }/g' include/internal/cpuid_x86.h
        sedinplace 's/set(PROCESSOR_IS_MIPS FALSE)/set(PROCESSOR_IS_MIPS TRUE)/g' CMakeLists.txt
        sedinplace 's/set(PROCESSOR_IS_X86 TRUE)/set(PROCESSOR_IS_X86 FALSE)/g' CMakeLists.txt
        CC="gcc -mabi=64 -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        CC="clang -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        make -j $MAKEJ
        make install
        ;;
    windows-x86)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE=" /MT /O2 /Ob2 /DNDEBUG" -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        nmake
        nmake install
        ;;
    windows-x86_64)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE=" /MT /O2 /Ob2 /DNDEBUG" -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_INSTALL_LIBDIR="lib" .
        nmake
        nmake install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cp -a include/* ../include/cpu_features/
sedinplace 's/cpu_features_macros.h/cpu_features\/cpu_features_macros.h/g' ../include/cpu_features/internal/hwcaps.h

cd ../..
