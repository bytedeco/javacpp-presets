#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" opencv
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    OPENCV_VERSION=3.0.0-beta
    [[ $PLATFORM == *64 ]] && BITS=x64 || BITS=x86
    download http://downloads.sourceforge.net/project/opencvlibrary/opencv-win/$OPENCV_VERSION/opencv-$OPENCV_VERSION.exe opencv-$OPENCV_VERSION.exe

    mkdir -p $PLATFORM
    cd $PLATFORM
    7za x -y ../opencv-$OPENCV_VERSION.exe opencv/build/OpenCV* opencv/build/include opencv/build/$BITS/vc10/lib opencv/build/$BITS/vc10/bin
    cd opencv
else
    OPENCV_VERSION=3.0.0-beta
    download https://github.com/Itseez/opencv/archive/$OPENCV_VERSION.tar.gz opencv-$OPENCV_VERSION.tar.gz
    download https://github.com/Itseez/opencv_contrib/archive/$OPENCV_VERSION.tar.gz opencv_contrib-$OPENCV_VERSION.tar.gz

    mkdir -p $PLATFORM
    cd $PLATFORM
    tar -xzvf ../opencv-$OPENCV_VERSION.tar.gz
    tar -xzvf ../opencv_contrib-$OPENCV_VERSION.tar.gz
    cd opencv-$OPENCV_VERSION
fi

case $PLATFORM in
    android-arm)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.6 -DCMAKE_INSTALL_PREFIX=.. -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/armeabi-v7a/* ../lib
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_TOOLCHAIN_NAME=x86-4.6 -DOPENCV_EXTRA_C_FLAGS="-mtune=atom -mssse3 -mfpmath=sse" -DOPENCV_EXTRA_CXX_FLAGS="-mtune=atom -mssse3 -mfpmath=sse" -DCMAKE_INSTALL_PREFIX=.. -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/x86/* ../lib
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DENABLE_SSE3=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install/strip
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install/strip
        ;;
    macosx-*)
        patch -Np1 < ../../../opencv-$OPENCV_VERSION-macosx.patch
        $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-headerpad_max_install_names"
        make -j4
        make install/strip
        ;;
    windows-x86)
        cp -r build/include ..
        cp -r build/x86/vc10/lib ..
        cp -r build/x86/vc10/bin ..
        ;;
    windows-x86_64)
        cp -r build/include ..
        cp -r build/x64/vc10/lib ..
        cp -r build/x64/vc10/bin ..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
