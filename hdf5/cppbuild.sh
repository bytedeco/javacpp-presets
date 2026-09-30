#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" hdf5
    popd
    exit
fi

ZLIB=zlib-1.3.2
HDF5_VERSION=2.2.0
AEC_VERSION=1.1.2
# zlib.net only serves the current latest release at its plain URL, so pinning a specific
# version there breaks again the moment a newer one ships (as already happened once for
# 1.3.1). Use zlib's own GitHub Release asset instead, which is stable per-version.
download "https://github.com/madler/zlib/releases/download/v${ZLIB#zlib-}/$ZLIB.tar.gz" $ZLIB.tar.gz
# support.hdfgroup.org's legacy FTP-style mirror no longer serves releases past 1.14.3;
# HDF Group now publishes source tarballs as GitHub Release assets instead.
download "https://github.com/HDFGroup/hdf5/releases/download/$HDF5_VERSION/hdf5-$HDF5_VERSION.tar.gz" hdf5-$HDF5_VERSION.tar.gz
# Use Github mirror repo rather than Gitlab repo for download speed
#download "https://gitlab.dkrz.de/k202009/libaec/uploads/45b10e42123edd26ab7b3ad92bcf7be2/libaec-$AEC_VERSION.tar.gz" libaec-$AEC_VERSION.tar.gz
download "https://github.com/MathisRosenhauer/libaec/releases/download/v$AEC_VERSION/libaec-$AEC_VERSION.tar.gz" libaec-$AEC_VERSION.tar.gz

mkdir -p $PLATFORM
pushd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xf ../hdf5-$HDF5_VERSION.tar.gz
tar --totals -xf ../libaec-$AEC_VERSION.tar.gz
tar --totals -xf ../$ZLIB.tar.gz
pushd hdf5-$HDF5_VERSION

#sedinplace '/cmake_minimum_required/d' $(find ./ -iname CMakeLists.txt)
sedinplace 's/# *cmakedefine/#cmakedefine/g' src/H5pubconf.h.in
sedinplace 's/COMPATIBILITY SameMinorVersion/COMPATIBILITY AnyNewerVersion/g' CMakeInstallation.cmake
sedinplace '/C_RUN (/{N;N;d;}' config/ConfigureChecks.cmake

# As of 1.14.0 the integrated cmake process for building aec/szip is broken
# Revisit integrated szip build with 1.14.1

# HDF5 2.x's CMake refuses JDKs older than 11, but Maven may be running on Java 8, so fall
# back to one of the newer JDKs that GitHub runners expose via JAVA_HOME_<version>_<arch>.
# The Java sources themselves still compile for Java 8 via the presets' own Maven build.
HDF5_JAVA_HOME=
for j in "${JAVA_HOME:-}" "${JAVA_HOME_11_X64:-}" "${JAVA_HOME_11_arm64:-}" "${JAVA_HOME_17_X64:-}" "${JAVA_HOME_17_arm64:-}" \
         "${JAVA_HOME_21_X64:-}" "${JAVA_HOME_21_arm64:-}" /usr/lib/jvm/*; do
    if [[ -n "$j" ]] && "$j/bin/java" -version 2>&1 | grep -qE 'version "(1[1-9]|[2-9][0-9])'; then
        HDF5_JAVA_HOME="$j"
        break
    fi
done
if [[ -z "$HDF5_JAVA_HOME" ]]; then
    # The containers used to cross-compile linux-armhf/linux-ppc64le/linux-x86
    # (ubuntu:bionic, centos:7) only ship JDK 8 via their own native package manager,
    # and none of the JAVA_HOME_<version>_<arch> runner variables above are visible
    # inside them either. HDF5's CMake only needs to run javac/jar on the build host
    # (not the cross-compile target), so grab a portable x86_64 JDK 11 as a last resort
    # rather than trying to match the C/C++ cross target's architecture.
    JDK11=jdk-11.0.2
    download "https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz" openjdk-11.0.2_linux-x64_bin.tar.gz
    tar --totals -xzf openjdk-11.0.2_linux-x64_bin.tar.gz
    HDF5_JAVA_HOME="$(pwd)/$JDK11"
fi
echo "Using JDK for HDF5's CMake Java build: ${HDF5_JAVA_HOME:-none found, leaving JAVA_HOME unchanged}"
HDF5_JAVA_HOME="${HDF5_JAVA_HOME:-${JAVA_HOME:-}}"

HDF5_CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_PREFIX_PATH="$INSTALL_PATH"
    -DBUILD_TESTING=OFF -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_JAVA=ON
    -DHDF5_ENABLE_ZLIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=ON -DHDF5_ENABLE_SZIP_ENCODING=ON -DSZIP_USE_EXTERNAL=OFF -DHDF5_USE_LIBAEC_STATIC=ON)

case $PLATFORM in
# The FAQ note that used to be here ("HDF5 does not currently support
# cross-compiling") is stale: as of HDF5 2.x's CMake, cross-compiling works fine for
# a C/C++/Java build (see linux-armhf/linux-ppc64le below and android-arm64/
# android-x86_64 further down) -- it just needs a plain toolchain file rather than
# the old autotools --host= invocations below. android-arm/android-x86 (32-bit) are
# left disabled: low utility today, and no currently-enabled precedent in this repo
# builds them either (openblas/opencv/ffmpeg keep these commented out too).
#    android-arm)
#        # Build libaec for szip first
#        mkdir -p ../libaec-$AEC_VERSION/build
#        pushd ../libaec-$AEC_VERSION/build
#        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
#        make -j $MAKEJ
#        make install
#        popd
#
#        patch -Np1 < ../../../hdf5-android.patch
#        ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" AR="$ANDROID_BIN-ar" RANLIB="$ANDROID_BIN-ranlib" CPP="$ANDROID_BIN-cpp" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/armeabi/include/ -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/armeabi/ -nostdlib -Wl,--fix-cortex-a8 -z text -L./" LIBS="-lgnustl_static -lgcc -ldl -lz -lm -lc" --enable-cxx --enable-java
#        make -j $MAKEJ
#        make install-strip
#        ;;
#    android-x86)
#        # Build libaec for szip first
#        mkdir -p ../libaec-$AEC_VERSION/build
#        pushd ../libaec-$AEC_VERSION/build
#        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
#        make -j $MAKEJ
#        make install
#        popd
#
#        patch -Np1 < ../../../hdf5-android.patch
#        ./configure --prefix=$INSTALL_PATH --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" AR="$ANDROID_BIN-ar" RANLIB="$ANDROID_BIN-ranlib" CPP="$ANDROID_BIN-cpp" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/x86/include/ -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/x86/ -nostdlib -z text -L." LIBS="-lgnustl_static -lgcc -ldl -lz -lm -lc" --enable-cxx --enable-java
#        make -j $MAKEJ
#        make install-strip
#        ;;
    android-arm64|android-x86_64)
        # PLATFORM_ROOT is the Android NDK root, set via the -Djavacpp.platform.root
        # Maven property that deploy-ubuntu/deploy-centos already export for every
        # android-* job (see the CI_DEPLOY_PLATFORM == android-* branch in those
        # actions); opencv's/openblas's own cppbuild.sh rely on the same variable for
        # their already-working android-arm64/android-x86_64 jobs.
        case $PLATFORM in
            android-arm64) ANDROID_ABI=arm64-v8a ;;
            android-x86_64) ANDROID_ABI=x86_64 ;;
        esac
        ANDROID_CMAKE_FLAGS=(-DCMAKE_TOOLCHAIN_FILE="${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake" -DANDROID_ABI="$ANDROID_ABI" -DANDROID_NATIVE_API_LEVEL=24)

        # Build libaec for szip first, with the same NDK toolchain as the main build
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" "${ANDROID_CMAKE_FLAGS[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        mkdir -p build
        pushd build
        JAVA_HOME="$HDF5_JAVA_HOME" "$CMAKE" "${ANDROID_CMAKE_FLAGS[@]}" "${HDF5_CMAKE_FLAGS[@]}" ..
        make -j $MAKEJ
        make install/strip
        popd
        ;;
    linux-armhf)
        # HDF5 2.x has no autotools build anymore, and its own CMake already degrades
        # gracefully when cross-compiling (H5ConversionTests falls back to documented
        # defaults when CMAKE_CROSSCOMPILING is set and no CMAKE_CROSSCOMPILING_EMULATOR
        # is given -- see config/ConfigureChecks.cmake), so a plain toolchain file is all
        # that's needed; no version-specific patch (the old hdf5-linux-armhf.patch was
        # written against HDF5 1.12.2's build tree, long before CMake supported this).
        ARMHF_CMAKE_FLAGS=()
        MACHINE_TYPE=$( uname -m )
        if [[ ! "$MACHINE_TYPE" =~ arm ]]; then
          echo "Not native arm so cross-compiling with arm-linux-gnueabihf"
          cat > arm.cmake <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
EOF
          ARMHF_CMAKE_FLAGS=(-DCMAKE_TOOLCHAIN_FILE="$(pwd)/arm.cmake")
        fi

        # Build libaec for szip first, with the same (native or cross) toolchain as HDF5
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" "${ARMHF_CMAKE_FLAGS[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        mkdir -p build
        pushd build
        JAVA_HOME="$HDF5_JAVA_HOME" "$CMAKE" "${ARMHF_CMAKE_FLAGS[@]}" "${HDF5_CMAKE_FLAGS[@]}" ..
        make -j $MAKEJ
        make install/strip
        popd
        ;;
    linux-arm64)
        # Build libaec for szip first
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        # Built natively on an arm64 runner; HDF5 2.x has no autotools build anymore
        mkdir -p build
        pushd build
        JAVA_HOME="$HDF5_JAVA_HOME" "$CMAKE" "${HDF5_CMAKE_FLAGS[@]}" ..
        make -j $MAKEJ
        make install/strip
        popd
        ;;
    linux-x86)
        # Build libaec for szip first
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_C_FLAGS="-m32" ..
        make -j $MAKEJ
        make install
        popd

        mkdir -p build
        pushd build
        JAVA_HOME="$HDF5_JAVA_HOME" "$CMAKE" "${HDF5_CMAKE_FLAGS[@]}" -DCMAKE_C_FLAGS="-m32" -DCMAKE_CXX_FLAGS="-m32" ..
        make -j $MAKEJ
        make install/strip
        popd
        ;;
    linux-x86_64)
        # Build libaec for szip first
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        mkdir -p build
        pushd build
        JAVA_HOME="$HDF5_JAVA_HOME" "$CMAKE" "${HDF5_CMAKE_FLAGS[@]}" ..
        make -j $MAKEJ
        make install/strip
        popd
        ;;
    linux-ppc64le)
        # Same rationale as linux-armhf above: no autotools build in HDF5 2.x, and its
        # CMake already has a graceful cross-compiling fallback, so a plain toolchain
        # file replaces the old hdf5-linux-ppc64le.patch (written against 1.12.2).
        PPC64LE_CMAKE_FLAGS=()
        MACHINE_TYPE=$( uname -m )
        if [[ ! "$MACHINE_TYPE" =~ ppc64 ]]; then
          echo "Not native ppc so cross-compiling with powerpc64le-linux-gnu"
          cat > ppc.cmake <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)
set(CMAKE_C_COMPILER powerpc64le-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER powerpc64le-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
EOF
          PPC64LE_CMAKE_FLAGS=(-DCMAKE_TOOLCHAIN_FILE="$(pwd)/ppc.cmake")
        fi

        # Build libaec for szip first, with the same (native or cross) toolchain as HDF5
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" "${PPC64LE_CMAKE_FLAGS[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        mkdir -p build
        pushd build
        JAVA_HOME="$HDF5_JAVA_HOME" "$CMAKE" "${PPC64LE_CMAKE_FLAGS[@]}" "${HDF5_CMAKE_FLAGS[@]}" ..
        make -j $MAKEJ
        make install/strip
        popd
        ;;
    macosx-*)
        # Build libaec for szip first
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        mkdir -p build
        pushd build
        JAVA_HOME="$HDF5_JAVA_HOME" "$CMAKE" "${HDF5_CMAKE_FLAGS[@]}" ..
        make -j $MAKEJ
        make install/strip
        popd
        ;;
    windows-x86)
        export CC="cl.exe"
        export CXX="cl.exe"

        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        ninja -j $MAKEJ
        ninja install
        popd

        mkdir -p ../$ZLIB/build
        pushd ../$ZLIB/build
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DZLIB_BUILD_TESTING=OFF ..
        ninja -j $MAKEJ
        ninja install
        popd

        mkdir -p build/bin
        cp ../lib/*.lib build/bin
        pushd build
        JAVA_HOME="$HDF5_JAVA_HOME" "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DZLIB_LIBRARY="$INSTALL_PATH/lib/zs.lib" -DZLIB_INCLUDE_DIR="$INSTALL_PATH/include" -DZLIB_USE_EXTERNAL=OFF -DSZIP_LIBRARY="$INSTALL_PATH/lib/szip-static.lib" -DSZIP_INCLUDE_DIR="$INSTALL_PATH/include" -DSZIP_USE_EXTERNAL=OFF -DHDF5_ENABLE_ZLIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=ON -DHDF5_ENABLE_SZIP_ENCODING=ON -DHDF5_USE_LIBAEC_STATIC=ON -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_JAVA=ON ..
        ninja -j $MAKEJ
        ninja install
        popd
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"

        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        ninja -j $MAKEJ
        ninja install
        popd

        mkdir -p ../$ZLIB/build
        pushd ../$ZLIB/build
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DZLIB_BUILD_TESTING=OFF ..
        ninja -j $MAKEJ
        ninja install
        popd

        mkdir -p build/bin
        cp ../lib/*.lib build/bin
        pushd build
        JAVA_HOME="$HDF5_JAVA_HOME" "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DZLIB_LIBRARY="$INSTALL_PATH/lib/zs.lib" -DZLIB_INCLUDE_DIR="$INSTALL_PATH/include" -DZLIB_USE_EXTERNAL=OFF -DSZIP_LIBRARY="$INSTALL_PATH/lib/szip-static.lib" -DSZIP_INCLUDE_DIR="$INSTALL_PATH/include" -DSZIP_USE_EXTERNAL=OFF -DHDF5_ENABLE_ZLIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=ON -DHDF5_ENABLE_SZIP_ENCODING=ON -DHDF5_USE_LIBAEC_STATIC=ON -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_JAVA=ON ..
        ninja -j $MAKEJ
        ninja install
        popd
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

[ -d "../java" ] && rm -r ../java
# HDF5 2.x moved the JNI-based Java API from java/src to java/src-jni (java/hdf is now a
# separate FFM-based implementation), and generates H5Version.java into the build tree.
cp -r java/src-jni ../java
rm -r ../java/test
cp build/java/src-jni/hdf/hdf5lib/H5Version.java ../java/hdf/hdf5lib/

# Return to cppbuild directory
popd
