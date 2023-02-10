#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mamba
    popd
    exit
fi

MAMBA_VERSION=2023.02.09
download "https://github.com/mamba-org/mamba/archive/refs/tags/$MAMBA_VERSION.tar.gz" mamba-$MAMBA_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xf ../mamba-$MAMBA_VERSION.tar.gz
cd mamba-$MAMBA_VERSION

case $PLATFORM in
    *)
        "$CMAKE" -B ../build/ -DBUILD_LIBMAMBA=ON -DBUILD_SHARED=ON -DBUILD_MICROMAMBA=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH 
        cd ../build
        make -j8
        make install
        cd ..
        ;;
esac

cd ../../
