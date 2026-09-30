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
echo "Using JDK for HDF5's CMake Java build: ${HDF5_JAVA_HOME:-none found, leaving JAVA_HOME unchanged}"
HDF5_JAVA_HOME="${HDF5_JAVA_HOME:-${JAVA_HOME:-}}"

HDF5_CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_PREFIX_PATH="$INSTALL_PATH"
    -DBUILD_TESTING=OFF -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_JAVA=ON
    -DHDF5_ENABLE_ZLIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=ON -DHDF5_ENABLE_SZIP_ENCODING=ON -DSZIP_USE_EXTERNAL=OFF -DHDF5_USE_LIBAEC_STATIC=ON)

case $PLATFORM in
# HDF5 does not currently support cross-compiling:
# https://support.hdfgroup.org/HDF5/faq/compile.html
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
    linux-armhf)
        # Build libaec for szip first
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ arm ]]; then
          ./configure --prefix=$INSTALL_PATH CC="gcc" CXX="g++" --enable-cxx --enable-java
          make -j $MAKEJ
          make install-strip
        else
          echo "Not native arm so assume cross compiling"
          patch -Np1 < ../../../hdf5-linux-armhf.patch || true
          #need this to run twice, first run fails so we fake the exit code too
          for x in 1 2; do
              "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=`pwd`/arm.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE" -DCMAKE_C_FLAGS="-D_GNU_SOURCE" -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_JAVA=ON . || true
          done
          make -j $MAKEJ
          make install
        fi
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
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          # Build libaec for szip first
          mkdir -p ../libaec-$AEC_VERSION/build
          pushd ../libaec-$AEC_VERSION/build
          "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
          make -j $MAKEJ
          make install
          popd

          ./configure --prefix=$INSTALL_PATH CC="gcc -m64" CXX="g++ -m64" --enable-cxx --enable-java --with-szlib
          make -j $MAKEJ
          make install-strip
        else
          echo "Not native ppc so assume cross compiling"
          patch -Np1 < ../../../hdf5-linux-ppc64le.patch || true
          #need this to run twice, first run fails so we fake the exit code too
          for x in 1 2; do
              "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=`pwd`/ppc.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE" -DCMAKE_C_FLAGS="-D_GNU_SOURCE" -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DSZAEC_TGZ_NAME:STRING="libaec-$AEC_VERSION.tar.gz" -DHDF5_ENABLE_SZIP_SUPPORT=ON -DHDF5_ENABLE_SZIP_ENCODING=ON -DUSE_LIBAEC=ON -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_JAVA=ON . || true
          done
          make -j $MAKEJ
          make install
        fi
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
