#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" caffe
    popd
    exit
fi

CAFFE_VERSION=master
download https://github.com/BVLC/caffe/archive/master.zip caffe-$CAFFE_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
unzip ../caffe-$CAFFE_VERSION.zip
cd caffe-$CAFFE_VERSION
cp Makefile.config.example Makefile.config

case $PLATFORM in
    linux-x86)
        CPU_ONLY=1 CC="gcc -m32" CXX="g++ -m32" BLAS=open DISTRIBUTE_DIR=.. make -j4 -e distribute
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" BLAS=open DISTRIBUTE_DIR=.. make -j4 -e distribute
        ;;
    macosx-*)
        CC="gcc -undefined dynamic_lookup" CXX="g++ -undefined dynamic_lookup" DISTRIBUTE_DIR=.. make --ignore-errors -j4 -e distribute
        mv ../lib/libcaffe.so ../lib/libcaffe.dylib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
