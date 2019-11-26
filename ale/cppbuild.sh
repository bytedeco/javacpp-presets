#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" ale
    popd
    exit
fi

ALE_VERSION=0.6.1
download https://github.com/mgbellemare/Arcade-Learning-Environment/archive/v$ALE_VERSION.tar.gz ale-v$ALE_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
echo "Decompressing archives..."
tar --totals -xzf ../ale-v$ALE_VERSION.tar.gz
cd Arcade-Learning-Environment-$ALE_VERSION

case $PLATFORM in
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
        make -j $MAKEJ
        cp -r src/* ../include
        cp libale.so ../lib
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
        make -j $MAKEJ
        cp -r src/* ../include
        cp libale.so ../lib
        ;;
    macosx-*)
        $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON -DCMAKE_MACOSX_RPATH=ON .
        make -j $MAKEJ
        cp -r src/* ../include
        cp libale.so ../lib
        ;;
    windows-x86)
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=.. -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread" .
        make -j $MAKEJ
        cp -r src/* ../include
        cp libale.dll ../lib
        ;;
    windows-x86_64)
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=.. -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread" .
        make -j $MAKEJ
        cp -r src/* ../include
        cp libale.dll ../lib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
