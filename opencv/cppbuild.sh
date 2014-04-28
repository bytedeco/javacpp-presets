if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

if [[ $PLATFORM == windows* ]]; then
    OPENCV_VERSION=2.4.9
    download http://downloads.sourceforge.net/project/opencvlibrary/opencv-win/$OPENCV_VERSION/opencv-$OPENCV_VERSION.exe opencv-$OPENCV_VERSION.exe

    INSTALL_DIR=/C/
else
    OPENCV_VERSION=2.4.9
    download https://github.com/Itseez/opencv/archive/$OPENCV_VERSION.tar.gz opencv-$OPENCV_VERSION.tar.gz

    tar -xzvf opencv-$OPENCV_VERSION.tar.gz
    mkdir opencv-$OPENCV_VERSION/build_$PLATFORM
    cd opencv-$OPENCV_VERSION/build_$PLATFORM
fi

case $PLATFORM in
    android-arm)
        cmake -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DCMAKE_INSTALL_PREFIX="$ANDROID_NDK/../" -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TBB=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF ..
        make -j4
        make install/strip
        ;;
    android-x86)
        cmake -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=x86 -DOPENCV_EXTRA_C_FLAGS="-mtune=atom -mssse3 -mfpmath=sse" -DOPENCV_EXTRA_CXX_FLAGS="-mtune=atom -mssse3 -mfpmath=sse" -DCMAKE_INSTALL_PREFIX="$ANDROID_NDK/../" -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TBB=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF ..
        make -j4
        make install/strip
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" cmake -DENABLE_SSE3=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TBB=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DLIB_SUFFIX=32 ..
        make -j4
        sudo make install/strip
        ;;
    linux-x86_64)
        cmake -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TBB=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DLIB_SUFFIX=64 ..
        make -j4
        sudo make install/strip
        ;;
    macosx-x86_64)
        patch -Np1 -d .. < ../../../opencv-$OPENCV_VERSION-macosx-x86_64.patch
        cmake -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TBB=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-headerpad_max_install_names" ..
        make -j4
        sudo make install/strip
        VER=${OPENCV_VERSION:0:3}
        BADPATH=lib
        LIBS="/usr/local/lib/libtbb.dylib /usr/local/lib/libopencv_*.$VER.dylib"
        for f in $LIBS; do sudo install_name_tool $f -id @rpath/`basename $f` \
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
        7za x -y opencv-$OPENCV_VERSION.exe -o$INSTALL_DIR -x!opencv/build/x64 -x!opencv/build/x86/vc10/staticlib -x!opencv/build/x86/vc11 -x!opencv/build/x86/vc12
        ;;
    windows-x86_64)
        7za x -y opencv-$OPENCV_VERSION.exe -o$INSTALL_DIR -x!opencv/build/x86 -x!opencv/build/x64/vc10/staticlib -x!opencv/build/x64/vc11 -x!opencv/build/x64/vc12
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

if [[ $PLATFORM != windows* ]]; then
    cd ../..
fi
