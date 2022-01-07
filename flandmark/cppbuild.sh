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
patch --binary -Np1 < ../../../flandmark.patch || true

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
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-armeabi-v7a/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        make -j $MAKEJ flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    android-arm64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-arm64-v8a/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        make -j $MAKEJ flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-x86/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        make -j $MAKEJ flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    android-x86_64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=${PLATFORM_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_NATIVE_API_LEVEL=24 -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/abi-x86_64/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        make -j $MAKEJ flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        make -j $MAKEJ flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        make -j $MAKEJ flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-armhf)
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        make -j $MAKEJ flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-arm64)
        CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        make -j $MAKEJ flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        else
          CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        fi
        make -j $MAKEJ flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-mips64el)
        CC="gcc -mabi=64" CXX="g++ -mabi=64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        make -j $MAKEJ flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    macosx-*)
        CXX="g++ -fpermissive" $CMAKE -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/lib/cmake/opencv4/ -DCMAKE_CXX_FLAGS="-std=c++11" .
        make -j $MAKEJ flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    windows-x86)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH .
        nmake flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.lib ../lib
        ;;
    windows-x86_64)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH .
        nmake flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.lib ../lib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
