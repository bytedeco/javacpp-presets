#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" pytorchvision
    popd
    exit
fi

PYTORCH_VISION_VERSION=0.21.0
download https://github.com/pytorch/vision/archive/v$PYTORCH_VISION_VERSION.tar.gz vision-$PYTORCH_VISION_VERSION.tar.gz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`

mkdir -p lib

PYTORCH_PATH="$INSTALL_PATH/../../../pytorch/cppbuild/$PLATFORM/pytorch"

echo "PYTORCH_PATH $PYTORCH_PATH"

echo "Decompressing archives..."
tar --totals -xzf ../vision-$PYTORCH_VISION_VERSION.tar.gz


cd vision-$PYTORCH_VISION_VERSION


case $PLATFORM in
    linux-x86_64)
        mkdir -p build
        cd build
        $CMAKE .. -DCMAKE_PREFIX_PATH=$PYTORCH_PATH
        $CMAKE  --build .
        cp ./libtorchvision.so $INSTALL_PATH/lib
        ;;
    
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac


cd ../..
