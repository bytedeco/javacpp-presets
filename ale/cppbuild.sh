#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" ale
    popd
    exit
fi

ALE_VERSION=0.7.4
SDL_VERSION=2.0.16
download https://github.com/mgbellemare/Arcade-Learning-Environment/archive/v$ALE_VERSION.tar.gz ale-v$ALE_VERSION.tar.gz
download https://github.com/libsdl-org/SDL/archive/refs/tags/release-$SDL_VERSION.tar.gz SDL-release-$SDL_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
echo "Decompressing archives..."
tar --totals -xzf ../ale-v$ALE_VERSION.tar.gz
tar --totals -xzf ../SDL-release-$SDL_VERSION.tar.gz
cd Arcade-Learning-Environment-$ALE_VERSION

sedinplace 's/message(FATAL_ERROR/message(WARNING/g' CMakeLists.txt
sedinplace 's/UNIX AND//g' src/CMakeLists.txt

case $PLATFORM in
    linux-x86)
        cd ../SDL-release-$SDL_VERSION
        CC="gcc -m32" CXX="g++ -m32" ./configure --prefix=$INSTALL_PATH --disable-video_wayland
        make -j $MAKEJ
        make install
        cd ../Arcade-Learning-Environment-$ALE_VERSION
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DSDL_SUPPORT=ON -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PYTHON_LIB=OFF .
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        cd ../SDL-release-$SDL_VERSION
        CC="gcc -m64" CXX="g++ -m64" ./configure --prefix=$INSTALL_PATH --disable-video_wayland
        make -j $MAKEJ
        make install
        cd ../Arcade-Learning-Environment-$ALE_VERSION
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DSDL_SUPPORT=ON -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PYTHON_LIB=OFF .
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DSDL_SUPPORT=ON -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PYTHON_LIB=OFF -DCMAKE_MACOSX_RPATH=ON .
        make -j $MAKEJ
        make install
        install_name_tool -change /usr/local/opt/sdl2/lib/libSDL2-2.0.0.dylib @rpath/libSDL2-2.0.0.dylib ../lib/libale.dylib
        ;;
    windows-x86)
        CC="gcc -m32 -DWIN32" CXX="g++ -m32 -DWIN32" $CMAKE -DCMAKE_BUILD_TYPE=Release -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DSDL_SUPPORT=ON -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PYTHON_LIB=OFF .
        make -j $MAKEJ
        make install
        ;;
    windows-x86_64)
        CC="gcc -m64 -DWIN32" CXX="g++ -m64 -DWIN32" $CMAKE -DCMAKE_BUILD_TYPE=Release -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON -DSDL_SUPPORT=ON -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PYTHON_LIB=OFF .
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
