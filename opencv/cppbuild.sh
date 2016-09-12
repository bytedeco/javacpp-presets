#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" opencv
    popd
    exit
fi

OPENCV_VERSION=3.1.0
download https://github.com/Itseez/opencv/archive/$OPENCV_VERSION.tar.gz opencv-$OPENCV_VERSION.tar.gz
download https://github.com/Itseez/opencv_contrib/archive/$OPENCV_VERSION.tar.gz opencv_contrib-$OPENCV_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
tar -xzvf ../opencv-$OPENCV_VERSION.tar.gz
tar -xzvf ../opencv_contrib-$OPENCV_VERSION.tar.gz
cd opencv-$OPENCV_VERSION

BUILD_CONTRIB_X="-DBUILD_opencv_stereo=OFF -DBUILD_opencv_plot=OFF -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_aruco=OFF -DBUILD_opencv_adas=OFF -DBUILD_opencv_bgsegm=OFF -DBUILD_opencv_bioinspired=ON -DBUILD_opencv_ccalib=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn=ON -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=ON -DBUILD_opencv_latentsvm=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_matlab=OFF -DBUILD_opencv_optflow=ON -DBUILD_opencv_reg=OFF -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_surface_matching=OFF -DBUILD_opencv_text=OFF -DBUILD_opencv_tracking=OFF -DBUILD_opencv_xfeatures2d=ON -DBUILD_opencv_ximgproc=ON -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=OFF"

case $PLATFORM in
    android-arm)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9 -DCMAKE_INSTALL_PREFIX=.. -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/armeabi-v7a/* ../lib
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_TOOLCHAIN_NAME=x86-4.9 -DOPENCV_EXTRA_C_FLAGS="-mtune=atom -mssse3 -mfpmath=sse" -DOPENCV_EXTRA_CXX_FLAGS="-mtune=atom -mssse3 -mfpmath=sse" -DCMAKE_INSTALL_PREFIX=.. -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/x86/* ../lib
        ;;
    linux-x86)
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DENABLE_SSE3=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install/strip
        ;;
    linux-x86_64)
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install/strip
        ;;
    linux-arm)
        CC=arm-linux-gnueabi-"$OLDCC" CXX=arm-linux-gnueabi-"$OLDCXX" CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_OPENMP=OFF -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install
        ;;
    linux-armhf)
        CC=arm-linux-gnueabihf-"$OLDCC" CXX=arm-linux-gnueabihf-"$OLDCXX" CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_SYSTEM_PROCESSOR=armv6 -DBUILD_TESTS=OFF -DCMAKE_CXX_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard" -DCMAKE_C_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard" -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_OPENMP=OFF -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install
        ;;
    linux-ppc64le)
        patch -d ../opencv_contrib-$OPENCV_VERSION -Np1 < ../../../opencv_contrib-$OPENCV_VERSION-linux.patch
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install/strip
        ;;
    macosx-*)
        patch -Np1 < ../../../opencv-$OPENCV_VERSION-macosx.patch
        $CMAKE -DCMAKE_INSTALL_PREFIX=.. -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=OFF -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j4
        make install/strip
        ;;
    windows-x86)
        patch -Np1 < ../../../opencv-$OPENCV_VERSION-windows.patch
        cd ..
        cd opencv_contrib-$OPENCV_VERSION
        patch -Np1 < ../../../opencv-contrib-bioinspired-$OPENCV_VERSION-windows.patch
        cd ..
        cd opencv-$OPENCV_VERSION
        BUILD_X="-DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF"
        WITH_X="-DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=OFF -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF"
        "$CMAKE" -G "Visual Studio 12 2013" -DCMAKE_INSTALL_PREFIX=.. $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $BUILD_CONTRIB_X -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release
        cp -r ../x86/vc12/lib ..
        cp -r ../x86/vc12/bin ..
        ;;
    windows-x86_64)
        patch -Np1 < ../../../opencv-$OPENCV_VERSION-windows.patch
        cd ..
        cd opencv_contrib-$OPENCV_VERSION
        patch -Np1 < ../../../opencv-contrib-bioinspired-$OPENCV_VERSION-windows.patch
        cd ..
        cd opencv-$OPENCV_VERSION
        BUILD_X="-DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF"
        WITH_X="-DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=OFF -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF"
        "$CMAKE" -G "Visual Studio 12 2013 Win64" -DCMAKE_INSTALL_PREFIX=.. $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $BUILD_CONTRIB_X -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release
        cp -r ../x64/vc12/lib ..
        cp -r ../x64/vc12/bin ..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
