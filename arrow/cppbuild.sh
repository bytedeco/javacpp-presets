#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" arrow
    popd
    exit
fi

ARROW_VERSION=0.15.1
download https://www.apache.org/dist/arrow/arrow-$ARROW_VERSION/apache-arrow-$ARROW_VERSION.tar.gz apache-arrow-$ARROW_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../apache-arrow-$ARROW_VERSION.tar.gz
cd apache-arrow-$ARROW_VERSION/cpp

case $PLATFORM in
    linux-armhf)
        CC="arm-linux-gnueabihf-gcc" CXX="arm-linux-gnueabihf-g++ -std=c++11" "$CMAKE" -DCMAKE_C_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard" -DCMAKE_CXX_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard" -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_VERBOSE_THIRDPARTY_BUILD=ON -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-arm64)
        CC="aarch64-linux-gnu-gcc" CXX="aarch64-linux-gnu-g++ -std=c++11" "$CMAKE" -DCMAKE_C_FLAGS="-mabi=lp64" -DCMAKE_CXX_FLAGS="-std=c++11 -mabi=lp64" -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_VERBOSE_THIRDPARTY_BUILD=ON -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-ppc64le)
        CC="powerpc64le-linux-gnu-gcc" CXX="powerpc64le-linux-gnu-gcc++ -std=c++11" "$CMAKE" -DCMAKE_C_FLAGS="-m64" -DCMAKE_CXX_FLAGS="-std=c++11 -m64" -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_VERBOSE_THIRDPARTY_BUILD=ON -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -std=c++11 -m32" "$CMAKE" -DCMAKE_C_FLAGS="-m32" -DCMAKE_CXX_FLAGS="-std=c++11 -m32" -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_VERBOSE_THIRDPARTY_BUILD=ON -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -std=c++11 -m64" "$CMAKE" -DCMAKE_C_FLAGS="-m64" -DCMAKE_CXX_FLAGS="-std=c++11 -m64" -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_VERBOSE_THIRDPARTY_BUILD=ON -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-*)
        CC="clang" CXX="clang++" "$CMAKE" -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_VERBOSE_THIRDPARTY_BUILD=ON -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86)
        cd ../..
        "$CMAKE" -G "Visual Studio 15 2017" -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_VERBOSE_THIRDPARTY_BUILD=ON -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF apache-arrow-$ARROW_VERSION/cpp
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        cd apache-arrow-$ARROW_VERSION/cpp
        ;;
    windows-x86_64)
        cd ../..
        "$CMAKE" -G "Visual Studio 15 2017 Win64" -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_VERBOSE_THIRDPARTY_BUILD=ON -DARROW_RPATH_ORIGIN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DARROW_BUILD_UTILITIES=OFF apache-arrow-$ARROW_VERSION/cpp
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        cd apache-arrow-$ARROW_VERSION/cpp
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../../..
