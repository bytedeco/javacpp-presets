#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" opencv
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    OPENCV_VERSION=2.4.9
    [[ $PLATFORM == *64 ]] && BITS=x64 || BITS=x86
    download http://downloads.sourceforge.net/project/opencvlibrary/opencv-win/$OPENCV_VERSION/opencv-$OPENCV_VERSION.exe opencv-$OPENCV_VERSION.exe

    mkdir -p $PLATFORM
    cd $PLATFORM
    7za x -y ../opencv-$OPENCV_VERSION.exe opencv/build/OpenCV* opencv/build/include opencv/build/$BITS/vc10/lib opencv/build/$BITS/vc10/bin
    cd opencv
else
    OPENCV_VERSION=2.4.9
    download https://github.com/Itseez/opencv/archive/$OPENCV_VERSION.tar.gz opencv-$OPENCV_VERSION.tar.gz

    mkdir -p $PLATFORM
    cd $PLATFORM
    tar -xzf ../opencv-$OPENCV_VERSION.tar.gz
    cd opencv-$OPENCV_VERSION
fi

case $PLATFORM in
    android-arm)
        cmake -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DCMAKE_INSTALL_PREFIX=.. -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TBB=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF
        make -j$NCPUS
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/armeabi-v7a/* ../lib
        ;;
    android-x86)
        cmake -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=x86 -DOPENCV_EXTRA_C_FLAGS="-mtune=atom -mssse3 -mfpmath=sse" -DOPENCV_EXTRA_CXX_FLAGS="-mtune=atom -mssse3 -mfpmath=sse" -DCMAKE_INSTALL_PREFIX=.. -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TBB=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF
        make -j$NCPUS
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/x86/* ../lib
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" cmake -DCMAKE_INSTALL_PREFIX=.. -DENABLE_SSE3=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TBB=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF
        make -j$NCPUS
        make install/strip
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" cmake -DCMAKE_INSTALL_PREFIX=.. -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TBB=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF
        make -j$NCPUS
        make install/strip
        ;;
    macosx-*)
        patch -Np1 < ../../../opencv-$OPENCV_VERSION-macosx.patch
        cmake -DCMAKE_INSTALL_PREFIX=.. -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TBB=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-headerpad_max_install_names"
        make -j$NCPUS
        make install/strip
        VER=${OPENCV_VERSION:0:3}
        BADPATH=lib
        LIBS="../lib/libtbb.dylib ../lib/libopencv_*.$VER.dylib"
        for f in $LIBS; do install_name_tool $f -id @rpath/`basename $f` \
            -add_rpath /usr/local/lib/ -add_rpath /opt/local/lib/ -add_rpath @loader_path/. \
            -change libtbb.dylib @rpath/libtbb.dylib \
            -change $BADPATH/libopencv_core.$VER.dylib @rpath/libopencv_core.$VER.dylib \
            -change $BADPATH/libopencv_calib3d.$VER.dylib @rpath/libopencv_calib3d.$VER.dylib \
            -change $BADPATH/libopencv_features2d.$VER.dylib @rpath/libopencv_features2d.$VER.dylib \
            -change $BADPATH/libopencv_flann.$VER.dylib @rpath/libopencv_flann.$VER.dylib \
            -change $BADPATH/libopencv_gpu.$VER.dylib @rpath/libopencv_gpu.$VER.dylib \
            -change $BADPATH/libopencv_highgui.$VER.dylib @rpath/libopencv_highgui.$VER.dylib \
            -change $BADPATH/libopencv_imgproc.$VER.dylib @rpath/libopencv_imgproc.$VER.dylib \
            -change $BADPATH/libopencv_legacy.$VER.dylib @rpath/libopencv_legacy.$VER.dylib \
            -change $BADPATH/libopencv_ml.$VER.dylib @rpath/libopencv_ml.$VER.dylib \
            -change $BADPATH/libopencv_nonfree.$VER.dylib @rpath/libopencv_nonfree.$VER.dylib \
            -change $BADPATH/libopencv_objdetect.$VER.dylib @rpath/libopencv_objdetect.$VER.dylib \
            -change $BADPATH/libopencv_photo.$VER.dylib @rpath/libopencv_photo.$VER.dylib \
            -change $BADPATH/libopencv_video.$VER.dylib @rpath/libopencv_video.$VER.dylib; done
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
