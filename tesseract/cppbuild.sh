#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tesseract
    popd
    exit
fi

TESSERACT_VERSION=5.5.2
download https://github.com/tesseract-ocr/tesseract/archive/$TESSERACT_VERSION.tar.gz tesseract-$TESSERACT_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../tesseract-$TESSERACT_VERSION.tar.gz
cd tesseract-$TESSERACT_VERSION

# Fix build on Mac
mv VERSION VERSION.txt
sedinplace 's/"VERSION"/"VERSION.txt"/g' CMakeLists.txt

# Disable external dependencies on asciidoc, libarchive, libtiff, etc
sedinplace '/  FATAL_ERROR/d' CMakeLists.txt
sedinplace '/find_package(TIFF)/d' CMakeLists.txt
sedinplace '/pkg_check_modules(TIFF/d' CMakeLists.txt
sedinplace '/NEON_COMPILE_FLAGS "-mfpu=neon"/d' CMakeLists.txt
sedinplace 's/if(COMPILER_SUPPORTS_MARCH_NATIVE)/if(FALSE)/g' CMakeLists.txt
sedinplace '/find_package(CpuFeaturesNdkCompat/,/CpuFeatures::ndk_compat)/d' CMakeLists.txt

LEPTONICA_PATH=$INSTALL_PATH/../../../leptonica/cppbuild/$PLATFORM/

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -d "$P/include/leptonica" ]]; then
            LEPTONICA_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

LEPTONICA_PATH="${LEPTONICA_PATH//\\//}"

export PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/

CMAKE_CONFIG="-DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$LEPTONICA_PATH -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR=$INSTALL_PATH/lib -DDISABLE_ARCHIVE=ON -DDISABLE_CURL=ON -DMARCH_NATIVE_OPT=OFF -DOPENMP_BUILD=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_TRAINING_TOOLS=OFF -DLEPT_TIFF_RESULT=1"

patch -Np1 < ../../../tesseract.patch

case $PLATFORM in
    android-arm)
        patch -Np1 < ../../../tesseract-android.patch
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH $CMAKE_CONFIG -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' .
        make -j $MAKEJ
        make install/strip
        ;;
    android-arm64)
        patch -Np1 < ../../../tesseract-android.patch
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH $CMAKE_CONFIG -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' .
        make -j $MAKEJ
        make install/strip
        ;;
    android-x86)
        patch -Np1 < ../../../tesseract-android.patch
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH $CMAKE_CONFIG -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' .
        make -j $MAKEJ
        make install/strip
        ;;
    android-x86_64)
        patch -Np1 < ../../../tesseract-android.patch
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH $CMAKE_CONFIG -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-armhf)
        export CC="arm-linux-gnueabihf-gcc -fPIC"
        export CXX="arm-linux-gnueabihf-g++ -fPIC"
        $CMAKE -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 $CMAKE_CONFIG -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-arm64)
        export CC="aarch64-linux-gnu-gcc -fPIC"
        export CXX="aarch64-linux-gnu-g++ -fPIC"
        $CMAKE -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=arm64 $CMAKE_CONFIG -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-ppc64le)
        export CC="powerpc64le-linux-gnu-gcc -fPIC"
        export CXX="powerpc64le-linux-gnu-g++ -fPIC"
        $CMAKE -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=ppc64le $CMAKE_CONFIG -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86)
        export CC="gcc -m32 -fPIC"
        export CXX="g++ -m32 -fPIC"
        $CMAKE $CMAKE_CONFIG -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        export CC="gcc -m64 -fPIC"
        export CXX="g++ -m64 -fPIC"
        $CMAKE $CMAKE_CONFIG -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/' .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-arm64)
        export CC="clang -arch arm64 -fPIC"
        export CXX="clang++ -arch arm64 -fPIC"
        $CMAKE -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=arm64 $CMAKE_CONFIG -DCMAKE_MACOSX_RPATH=ON -DCMAKE_CXX_FLAGS='-Wl,-rpath,@loader_path/' .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        export CC="clang -arch x86_64 -fPIC"
        export CXX="clang++ -arch x86_64 -fPIC"
        $CMAKE $CMAKE_CONFIG -DCMAKE_MACOSX_RPATH=ON -DCMAKE_CXX_FLAGS='-Wl,-rpath,@loader_path/' .
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG -DSW_BUILD=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG -DSW_BUILD=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
