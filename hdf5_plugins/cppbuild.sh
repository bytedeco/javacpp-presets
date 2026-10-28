#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" hdf5_plugins
    popd
    exit
fi

HDF5_PLUGINS_VERSION=2.2.0
# The release tarball bundles the sources of zstd, lz4, bzip2, libjpeg, bitshuffle, c-blosc and
# c-blosc2 under libs/, so no other download is needed.
download "https://github.com/HDFGroup/hdf5_plugins/releases/download/$HDF5_PLUGINS_VERSION/hdf5_plugins-$HDF5_PLUGINS_VERSION.tar.gz" hdf5_plugins-$HDF5_PLUGINS_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../hdf5_plugins-$HDF5_PLUGINS_VERSION.tar.gz
cd hdf5_plugins-$HDF5_PLUGINS_VERSION

# Some plugins (blosc, blosc2, bshuf, jpeg) call HDF5 functions, so they link against the shared HDF5 library
# built by the hdf5 module. It must be loaded by the Loader before the plugins, see HDF5Plugins.load()
HDF5_PATH=$INSTALL_PATH/../../../hdf5/cppbuild/$PLATFORM/

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -f "$P/include/H5public.h" ]]; then
            HDF5_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

HDF5_PATH="${HDF5_PATH//\\//}"

# Plugins are modules loaded at runtime by HDF5, so they get their compression libraries linked in statically
CMAKE_CONFIG=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$HDF5_PATH" -DHDF5_ROOT="$HDF5_PATH" -DUSE_SHARED_LIBS=ON -DHDF5_USE_STATIC_LIBRARIES=OFF -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH"
    -DH5PL_ALLOW_EXTERNAL_SUPPORT=TGZ -DH5PL_BUILD_TESTING=OFF -DH5PL_BUILD_EXAMPLES=OFF -DH5PL_CPACK_ENABLE=OFF -DBUILD_SHARED_LIBS=ON
    -DENABLE_ZSTD=ON -DENABLE_LZ4=ON -DENABLE_JPEG=ON -DENABLE_BZIP2=ON -DENABLE_BSHUF=ON -DENABLE_BLOSC=ON -DENABLE_BLOSC2=ON
    -DENABLE_BITGROOM=ON -DENABLE_GRANULAR_BITROUND=ON -DENABLE_BITROUND=ON -DENABLE_LZF=ON -DENABLE_ZFP=ON)
ANDROID_CMAKE_FLAGS=(-DCMAKE_TOOLCHAIN_FILE="${PLATFORM_ROOT:-}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL=24
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH)

mkdir -p build
cd build

case $PLATFORM in
    android-arm64)
        "$CMAKE" "${ANDROID_CMAKE_FLAGS[@]}" -DANDROID_ABI=arm64-v8a "${CMAKE_CONFIG[@]}" ..
        make -j $MAKEJ
        make install/strip
        ;;
    android-x86_64)
        "$CMAKE" "${ANDROID_CMAKE_FLAGS[@]}" -DANDROID_ABI=x86_64 "${CMAKE_CONFIG[@]}" ..
        make -j $MAKEJ
        make install/strip
        ;;
    linux-arm64)
        export CC="gcc -fPIC"
        "$CMAKE" "${CMAKE_CONFIG[@]}" ..
        make -j $MAKEJ
        make install/strip
        ;;
    linux-ppc64le)
        export CC="powerpc64le-linux-gnu-gcc -fPIC"
        "$CMAKE" -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=ppc64le "${CMAKE_CONFIG[@]}" ..
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        export CC="gcc -m64 -fPIC"
        "$CMAKE" "${CMAKE_CONFIG[@]}" ..
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-arm64)
        export CC="clang -arch arm64 -fPIC"
        "$CMAKE" -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=arm64 "${CMAKE_CONFIG[@]}" ..
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        export CC="clang -arch x86_64 -fPIC"
        "$CMAKE" "${CMAKE_CONFIG[@]}" ..
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"
        # JPEG is disabled by upstream's CMake on Windows
        "$CMAKE" -G "Ninja" "${CMAKE_CONFIG[@]}" ..
        ninja -j $MAKEJ
        ninja install
        # Windows installs modules under bin/, but keep all plugins in the same directory as other platforms
        mkdir -p "$INSTALL_PATH/lib/plugin"
        mv "$INSTALL_PATH"/bin/h5*.dll "$INSTALL_PATH/lib/plugin/" || true
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

# ZFP also installs its static library and headers, which are not needed here
rm -f "$INSTALL_PATH"/lib/libh5z*.a "$INSTALL_PATH"/lib/h5z*.lib "$INSTALL_PATH"/include/H5Zzfp*.h

cd ../../..
