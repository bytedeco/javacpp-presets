#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" leptonica
    popd
    exit
fi

NASM_VERSION=2.14
ZLIB=zlib-1.2.12
GIFLIB=giflib-5.2.1
LIBJPEG=libjpeg-turbo-2.1.3
LIBPNG=libpng-1.6.37 # warning: libpng16 doesn't work on CentOS 6 for some reason
LIBTIFF=tiff-4.3.0
LIBWEBP=libwebp-1.2.2
LEPTONICA_VERSION=1.82.0
download https://download.videolan.org/contrib/nasm/nasm-$NASM_VERSION.tar.gz nasm-$NASM_VERSION.tar.gz
download http://zlib.net/$ZLIB.tar.gz $ZLIB.tar.gz
download http://downloads.sourceforge.net/project/giflib/$GIFLIB.tar.gz $GIFLIB.tar.gz
download http://downloads.sourceforge.net/project/libjpeg-turbo/2.1.3/$LIBJPEG.tar.gz $LIBJPEG.tar.gz
download https://sourceforge.net/projects/libpng/files/libpng16/1.6.37/$LIBPNG.tar.gz $LIBPNG.tar.gz
download http://download.osgeo.org/libtiff/$LIBTIFF.tar.gz $LIBTIFF.tar.gz
download http://downloads.webmproject.org/releases/webp/$LIBWEBP.tar.gz $LIBWEBP.tar.gz
download https://github.com/DanBloomberg/leptonica/releases/download/$LEPTONICA_VERSION/leptonica-$LEPTONICA_VERSION.tar.gz leptonica-$LEPTONICA_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../nasm-$NASM_VERSION.tar.gz
tar --totals -xzf ../$ZLIB.tar.gz
tar --totals -xzf ../$GIFLIB.tar.gz
tar --totals -xzf ../$LIBJPEG.tar.gz
tar --totals -xzf ../$LIBPNG.tar.gz
tar --totals -xzf ../$LIBTIFF.tar.gz
tar --totals -xzf ../$LIBWEBP.tar.gz
tar --totals -xzf ../leptonica-$LEPTONICA_VERSION.tar.gz

# https://src.fedoraproject.org/rpms/giflib/blob/rawhide/f/CMakeLists.txt
# https://sourceforge.net/p/giflib/feature-requests/6/
patch -Np1 -d $GIFLIB < ../../giflib.patch || true

sedinplace 's/add_library(zlib SHARED/add_library(zlib STATIC/g' $ZLIB/CMakeLists.txt
sedinplace 's/add_library(giflib SHARED/add_library(giflib STATIC/g' $GIFLIB/CMakeLists.txt
sedinplace 's/if(WIN32)/if(FALSE)/g' $GIFLIB/CMakeLists.txt
sedinplace 's/include(OpenGLChecks)/set(HAVE_OPENGL FALSE)/g' $LIBTIFF/CMakeLists.txt
sedinplace 's/SOVERSION 5..../SOVERSION 5/g' leptonica-$LEPTONICA_VERSION/src/CMakeLists.txt
sedinplace 's/VERSION   ${VERSION_PLAIN}/VERSION   5/g' leptonica-$LEPTONICA_VERSION/src/CMakeLists.txt
sedinplace 's/leptonica-${VERSION_PLAIN}/leptonica-5/g' leptonica-$LEPTONICA_VERSION/src/CMakeLists.txt
sedinplace 's/message(FATAL_ERROR/message(WARNING/g' leptonica-$LEPTONICA_VERSION/CMakeLists.txt

cd nasm-$NASM_VERSION
# fix for build with GCC 8.x
sedinplace 's/void pure_func/void/g' include/nasmlib.h
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ V=0
make install
cd ..

export PATH=$INSTALL_PATH/bin:$PATH
export PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig/

CMAKE_CONFIG="-DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$INSTALL_PATH -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_INSTALL_LIBDIR=$INSTALL_PATH/lib -DBUILD_SHARED_LIBS=OFF -DENABLE_SHARED=FALSE -DPNG_SHARED=OFF"
WEBP_CONFIG="-DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_EXTRAS=OFF -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_WEBP_JS=OFF"

case $PLATFORM in
    android-arm)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export STRIP="$ANDROID_PREFIX-strip"
        export CFLAGS="-DS_IREAD=S_IRUSR -DS_IWRITE=S_IWUSR -pthread -I$INSTALL_PATH/include/ -L$INSTALL_PATH/lib/ $ANDROID_FLAGS"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-ldl -lm -lc"
        export CMAKE_CONFIG="-DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH $CMAKE_CONFIG"
        cd $ZLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        rm contrib/arm-neon/android-ndk.c || true
        $CMAKE $CMAKE_CONFIG -DPNG_ARM_NEON=off .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        patch -Np1 < ../../../libwebp-arm.patch
        $CMAKE $CMAKE_CONFIG $WEBP_CONFIG -DWEBP_ENABLE_SIMD=OFF .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    android-arm64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export STRIP="$ANDROID_PREFIX-strip"
        export CFLAGS="-DS_IREAD=S_IRUSR -DS_IWRITE=S_IWUSR -pthread -I$INSTALL_PATH/include/ -L$INSTALL_PATH/lib/ $ANDROID_FLAGS"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-ldl -lm -lc"
        export CMAKE_CONFIG="-DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH $CMAKE_CONFIG"
        cd $ZLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        rm contrib/arm-neon/android-ndk.c || true
        $CMAKE $CMAKE_CONFIG -DPNG_ARM_NEON=off .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        patch -Np1 < ../../../libwebp-arm.patch
        $CMAKE $CMAKE_CONFIG $WEBP_CONFIG -DWEBP_ENABLE_SIMD=OFF .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
     android-x86)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export STRIP="$ANDROID_PREFIX-strip"
        export CFLAGS="-DS_IREAD=S_IRUSR -DS_IWRITE=S_IWUSR -pthread -I$INSTALL_PATH/include/ -L$INSTALL_PATH/lib/ $ANDROID_FLAGS"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-ldl -lm -lc"
        export CMAKE_CONFIG="-DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH $CMAKE_CONFIG"
        cd $ZLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        sedinplace 's/if(CPU_TYPE STREQUAL "x86_64")/if(FALSE)/g' simd/CMakeLists.txt
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        rm contrib/arm-neon/android-ndk.c || true
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        $CMAKE $CMAKE_CONFIG $WEBP_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
     android-x86_64)
        export AR="$ANDROID_PREFIX-ar"
        export RANLIB="$ANDROID_PREFIX-ranlib"
        export STRIP="$ANDROID_PREFIX-strip"
        export CFLAGS="-DS_IREAD=S_IRUSR -DS_IWRITE=S_IWUSR -pthread -I$INSTALL_PATH/include/ -L$INSTALL_PATH/lib/ $ANDROID_FLAGS"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-ldl -lm -lc"
        export CMAKE_CONFIG="-DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH $CMAKE_CONFIG"
        cd $ZLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        sedinplace 's/if(CPU_TYPE STREQUAL "x86_64")/if(TRUE)/g' simd/CMakeLists.txt
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        rm contrib/arm-neon/android-ndk.c || true
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        $CMAKE $CMAKE_CONFIG $WEBP_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="gcc -m32 -fPIC"
        export CXX="g++ -m32 -fPIC"
        cd $ZLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        sedinplace 's/if(CPU_TYPE STREQUAL "x86_64")/if(FALSE)/g' simd/CMakeLists.txt
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        $CMAKE $CMAKE_CONFIG $WEBP_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="gcc -m64 -fPIC"
        export CXX="g++ -m64 -fPIC"
        cd $ZLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        sedinplace 's/if(CPU_TYPE STREQUAL "x86_64")/if(TRUE)/g' simd/CMakeLists.txt
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        $CMAKE $CMAKE_CONFIG $WEBP_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-armhf)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="arm-linux-gnueabihf-gcc -fPIC"
        export CXX="arm-linux-gnueabihf-g++ -fPIC"
        export STRIP="arm-linux-gnueabihf-strip"
        export CMAKE_CONFIG="-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv6 $CMAKE_CONFIG"
        cd $ZLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        rm contrib/arm-neon/android-ndk.c || true
        $CMAKE $CMAKE_CONFIG -DPNG_ARM_NEON=off .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        patch -Np1 < ../../../libwebp-arm.patch
        $CMAKE $CMAKE_CONFIG $WEBP_CONFIG -DWEBP_ENABLE_SIMD=OFF .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-arm64)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="aarch64-linux-gnu-gcc -fPIC"
        export CXX="aarch64-linux-gnu-g++ -fPIC"
        export STRIP="aarch64-linux-gnu-strip"
        export CMAKE_CONFIG="-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 $CMAKE_CONFIG"
        cd $ZLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        $CMAKE $CMAKE_CONFIG $WEBP_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-ppc64le)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="powerpc64le-linux-gnu-gcc -fPIC"
        export CXX="powerpc64le-linux-gnu-g++ -fPIC"
        export STRIP="powerpc64le-linux-gnu-strip"
        export CMAKE_CONFIG="-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=ppc64le $CMAKE_CONFIG"
        cd $ZLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        $CMAKE $CMAKE_CONFIG $WEBP_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-arm64)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="clang -arch arm64 -fPIC"
        export CXX="clang++ -arch arm64 -fPIC"
        export CMAKE_CONFIG="-DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=armv8 $CMAKE_CONFIG"
        cd $ZLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        $CMAKE $CMAKE_CONFIG -DPNG_ARM_NEON=off .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        $CMAKE $CMAKE_CONFIG $WEBP_CONFIG -DWEBP_ENABLE_SIMD=OFF .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF -DCMAKE_MACOSX_RPATH=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-x86_64)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="clang -arch x86_64 -fPIC"
        export CXX="clang++ -arch x86_64 -fPIC"
        cd $ZLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        $CMAKE $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        $CMAKE $CMAKE_CONFIG $WEBP_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF -DCMAKE_MACOSX_RPATH=ON .
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="gcc -m32"
        export CXX="g++ -m32"
        cd $ZLIB
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG $WEBP_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF -DSW_BUILD=OFF .
        make -j $MAKEJ
        make install
        ;;
    windows-x86_64)
        export CFLAGS="-pthread -I$INSTALL_PATH/include/"
        export CXXFLAGS="$CFLAGS"
        export CPPFLAGS="$CFLAGS"
        export LDFLAGS="-L$INSTALL_PATH/lib/"
        export CC="gcc -m64"
        export CXX="g++ -m64"
        cd $ZLIB
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$GIFLIB
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBPNG
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBWEBP
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG $WEBP_CONFIG .
        make -j $MAKEJ
        make install
        cd ../$LIBTIFF
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG -Dlzma=OFF -Dzstd=OFF .
        make -j $MAKEJ
        make install
        cd ../leptonica-$LEPTONICA_VERSION
        $CMAKE -G "MSYS Makefiles" $CMAKE_CONFIG -DBUILD_SHARED_LIBS=ON -DOPENJPEG_SUPPORT=OFF -DSW_BUILD=OFF .
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

# fix broken dependencies from files for cmake
sedinplace 's:bin/libleptonica:libleptonica:g' ../lib/cmake/leptonica/LeptonicaTargets-release.cmake
sedinplace 's:INTERFACE_LINK_LIBRARIES *".*"::g' ../lib/cmake/leptonica/LeptonicaConfig.cmake ../lib/cmake/leptonica/LeptonicaTargets.cmake
sedinplace 's:INTERFACE_INCLUDE_DIRECTORIES *".*":INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../../include":g' ../lib/cmake/leptonica/LeptonicaConfig.cmake ../lib/cmake/leptonica/LeptonicaTargets.cmake
sedinplace 's:Leptonica_INCLUDE_DIRS *".*":Leptonica_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../include/leptonica":g' ../lib/cmake/leptonica/LeptonicaConfig.cmake ../lib/cmake/leptonica/LeptonicaTargets.cmake

cd ../..
