#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" ale
    popd
    exit
fi

ALE_VERSION=5c7dfa5908a2bf8b1de354d0d9d44c9c3965abbb
download https://github.com/mgbellemare/Arcade-Learning-Environment/archive/$ALE_VERSION.tar.gz ale-$ALE_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
echo "Decompressing archives..."
tar --totals -xzf ../ale-$ALE_VERSION.tar.gz
cd Arcade-Learning-Environment-$ALE_VERSION

case $PLATFORM in
    linux-x86)
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON
        make -j $MAKEJ
        cp -r src/* ../include
        cp libale.so ../lib
        ;;
    linux-x86_64)
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON
        make -j $MAKEJ
        cp -r src/* ../include
        cp libale.so ../lib
        ;;
    macosx-*)
        $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON
        make -j $MAKEJ
        cp -r src/* ../include
        cp libale.so ../lib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
