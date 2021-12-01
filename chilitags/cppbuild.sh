#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" chilitags
    popd
    exit
fi

CHILITAGS_VERSION=master
download https://github.com/chili-epfl/chilitags/archive/$CHILITAGS_VERSION.zip chilitags-$CHILITAGS_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include/chilitags lib bin
unzip -o ../chilitags-$CHILITAGS_VERSION.zip
cd chilitags-$CHILITAGS_VERSION

OPENCV_PATH=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -d "$P/include/opencv2" ]]; then
            OPENCV_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

case $PLATFORM in
    android-arm)
        ANDROID_STANDALONE_TOOLCHAIN=$ANDROID_NDK ANDROID_NATIVE_API_LEVEL=24 ANDROID_ABI=armeabi-v7a ANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9 $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-armeabi-v7a/ .
        make -j $MAKEJ
        make install
        ;;
    android-arm64)
        ANDROID_STANDALONE_TOOLCHAIN=$ANDROID_NDK ANDROID_NATIVE_API_LEVEL=24 ANDROID_ABI=arm64-v8a ANDROID_TOOLCHAIN_NAME=aarch64-linux-android-4.9 $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-arm64-v8a/ .
        make -j $MAKEJ
        make install
        ;;
    android-x86)
        ANDROID_STANDALONE_TOOLCHAIN=$ANDROID_NDK ANDROID_NATIVE_API_LEVEL=24 ANDROID_ABI=x86 ANDROID_TOOLCHAIN_NAME=x86-4.9 $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-x86/ .
        make -j $MAKEJ
        make install
        ;;
    android-x86_64)
        ANDROID_STANDALONE_TOOLCHAIN=$ANDROID_NDK ANDROID_NATIVE_API_LEVEL=24 ANDROID_ABI=x86_64 ANDROID_TOOLCHAIN_NAME=x86_64-4.9 $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-x86_64/ .
        make -j $MAKEJ
        make install
        ;;
    linux-x86)
        CXX="g++ -m32 -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ .
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        CXX="g++ -m64 -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ .
        make -j $MAKEJ
        make install
        ;;
    linux-armhf)
        CXX="arm-linux-gnueabihf-g++ -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ -DCMAKE_CXX_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard -Wl,-allow-shlib-undefined" .
        make -j $MAKEJ
        make install
        ;;
    linux-arm64)
        CXX="aarch64-linux-gnu-g++ -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ .
        make -j $MAKEJ
        make install
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CXX="g++ -m64 -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ .
        else
          CXX="powerpc64le-linux-gnu-g++ -m64 -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ .
        fi
        make -j $MAKEJ
        make install
        ;;
    linux-mips64el)
        CXX="g++ -mabi=64 -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ .
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        CXX="clang++ -fPIC" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ .
        make -j $MAKEJ
        make install
        ;;
    windows-x86)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH .
        nmake chilitags_static
        cp *.hpp include/*.hpp ../include/chilitags/
        cp src/*.lib ../lib/
        ;;
    windows-x86_64)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$OPENCV_PATH .
        nmake chilitags_static
        cp *.hpp include/*.hpp ../include/chilitags/
        cp src/*.lib ../lib/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
