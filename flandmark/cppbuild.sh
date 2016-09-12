#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" flandmark
    popd
    exit
fi

FLANDMARK_VERSION=master
download https://github.com/uricamic/flandmark/archive/$FLANDMARK_VERSION.zip flandmark-$FLANDMARK_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
unzip -o ../flandmark-$FLANDMARK_VERSION.zip
cd flandmark-$FLANDMARK_VERSION

OPENCV_PATH=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/

case $PLATFORM in
    android-arm)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=$INSTALL_PATH/../../android-arm.cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/ -DANDROID_NDK_ABI_NAME=armeabi_v7a
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=$INSTALL_PATH/../../android-x86.cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/ -DANDROID_NDK_ABI_NAME=x86
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-x86)
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-x86_64)
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-armhf)
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-ppc64le)
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    macosx-*)
        CXX="g++ -fpermissive" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    windows-x86)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH
        nmake flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.lib ../lib
        ;;
    windows-x86_64)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH
        nmake flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.lib ../lib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
