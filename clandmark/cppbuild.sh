#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" clandmark
    popd
    exit
fi

CLANDMARK_VERSION=master
download https://github.com/uricamic/clandmark/archive/$CLANDMARK_VERSION.zip clandmark-$CLANDMARK_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
unzip -o ../clandmark-$CLANDMARK_VERSION.zip
cd clandmark-$CLANDMARK_VERSION

OPENCV_PATH=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/

case $PLATFORM in
    android-arm)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=$INSTALL_PATH/../../android-arm.cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/ -DANDROID_NDK_ABI_NAME=armeabi_v7a -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install clandmark
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=$INSTALL_PATH/../../android-x86.cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/ -DANDROID_NDK_ABI_NAME=x86 -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install clandmark
        ;;
    linux-x86)
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/ -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install clandmark
        ;;
    linux-x86_64)
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/ -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install clandmark
        ;;
    linux-armhf)
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/ -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install clandmark
        ;;
    macosx-*)
        CXX="g++ -fpermissive" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/ -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install/strip
        ;;
    windows-x86)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH -DCMAKE_INSTALL_PREFIX=..
        nmake clandmark
        nmake
        nmake install clandmark
        ;;
    windows-x86_64)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH -DCMAKE_INSTALL_PREFIX=..
        nmake
        nmake install clandmark
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
