#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" opencv
    popd
    exit
fi

OPENCV_VERSION=3.4.0
download https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz opencv-$OPENCV_VERSION.tar.gz
download https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.tar.gz opencv_contrib-$OPENCV_VERSION.tar.gz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../opencv-$OPENCV_VERSION.tar.gz
tar --totals -xzf ../opencv_contrib-$OPENCV_VERSION.tar.gz

cd opencv_contrib-$OPENCV_VERSION
patch -Np1 < ../../../opencv_contrib.patch

cd ../opencv-$OPENCV_VERSION
patch -Np1 < ../../../opencv.patch

# fixes for iOS
sed -i="" '/#if defined(NO)/a\
#undef NO\
' modules/stitching/include/opencv2/stitching/detail/exposure_compensate.hpp
patch -Np1 < ../../../opencv-ios.patch

BUILD_CONTRIB_X="-DBUILD_opencv_stereo=OFF -DBUILD_opencv_plot=ON -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_aruco=OFF -DBUILD_opencv_adas=OFF -DBUILD_opencv_bgsegm=OFF -DBUILD_opencv_bioinspired=ON -DBUILD_opencv_ccalib=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn_modern=OFF -DBUILD_opencv_freetype=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=ON -DBUILD_opencv_latentsvm=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_matlab=OFF -DBUILD_opencv_optflow=ON -DBUILD_opencv_reg=OFF -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_surface_matching=OFF -DBUILD_opencv_text=ON -DBUILD_opencv_tracking=ON -DBUILD_opencv_xfeatures2d=ON -DBUILD_opencv_ximgproc=ON -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=OFF -DWITH_LAPACK=OFF"

GPU_FLAGS="-DWITH_CUDA=OFF -DWITH_OPENCL=ON"
if [[ "$EXTENSION" == *gpu ]]; then
    GPU_FLAGS="-DWITH_CUDA=ON -DWITH_OPENCL=ON -DCUDA_VERSION=9.1 -DCUDA_ARCH_BIN=3.0 -DCUDA_ARCH_PTX=3.0 -DCUDA_NVCC_FLAGS=--expt-relaxed-constexpr"
fi

case $PLATFORM in
    android-arm)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9 -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON $GPU_FLAGS -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j $MAKEJ
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/armeabi-v7a/* ../lib
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_TOOLCHAIN_NAME=x86-4.9 -DOPENCV_EXTRA_C_FLAGS="-mtune=atom -mssse3 -mfpmath=sse" -DOPENCV_EXTRA_CXX_FLAGS="-mtune=atom -mssse3 -mfpmath=sse" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON $GPU_FLAGS -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j $MAKEJ
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/x86/* ../lib
        ;;
    ios-arm64)
        $CMAKE -GXcode -DCMAKE_TOOLCHAIN_FILE=platforms/ios/cmake/Toolchains/Toolchain-iPhoneOS_Xcode.cmake -DIOS_ARCH=arm64 -DAPPLE_FRAMEWORK=ON -DCMAKE_MACOSX_BUNDLE=ON -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO -DCMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE=NO -DBUILD_opencv_world=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=OFF $GPU_FLAGS -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        xcodebuild -arch arm64 -sdk iphoneos -configuration Release -parallelizeTargets -jobs $MAKEJ ONLY_ACTIVE_ARCH=NO -target ALL_BUILD build
        $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -P cmake_install.cmake
        cp ../share/OpenCV/3rdparty/lib/* ../lib
        ;;
    ios-x86_64)
        $CMAKE -GXcode -DCMAKE_TOOLCHAIN_FILE=platforms/ios/cmake/Toolchains/Toolchain-iPhoneSimulator_Xcode.cmake -DIOS_ARCH=x86_64 -DAPPLE_FRAMEWORK=ON -DCMAKE_MACOSX_BUNDLE=ON -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO -DCMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE=NO -DBUILD_opencv_world=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=OFF $GPU_FLAGS -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        xcodebuild -arch x86_64 -sdk iphonesimulator -configuration Release -parallelizeTargets -jobs $MAKEJ ONLY_ACTIVE_ARCH=NO -target ALL_BUILD build
        $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -P cmake_install.cmake
        cp ../share/OpenCV/3rdparty/lib/* ../lib
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DENABLE_SSE3=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules -DCMAKE_CXX_FLAGS="-w"
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j $MAKEJ
        make install/strip
        ;;
    linux-arm)
        CC="arm-linux-gnueabi-gcc" CXX="arm-linux-gnueabi-g++" CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF -DBUILD_opencv_python3=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_OPENMP=OFF $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j $MAKEJ
        make install
        ;;
    linux-armhf)
        CC="arm-linux-gnueabihf-gcc" CXX="arm-linux-gnueabihf-g++" CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_SYSTEM_PROCESSOR=armv6 -DBUILD_TESTS=OFF -DCMAKE_CXX_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard" -DCMAKE_C_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard" -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF -DBUILD_opencv_python3=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_OPENMP=ON $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j $MAKEJ
        make install/strip
        ;;
    linux-arm64)
        CC="aarch64-linux-gnu-gcc" CXX="aarch64-linux-gnu-g++" CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DAARCH64=ON -DENABLE_NEON=OFF -DENABLE_SSE=OFF -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DBUILD_TESTS=OFF -DCMAKE_CXX_FLAGS="" -DCMAKE_C_FLAGS="" -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_OPENMP=ON $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j $MAKEJ
        make install/strip
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        else
          echo "Not native ppc so assume cross compiling"
          PKG_CONFIG_PATH=/usr/lib/powerpc64le-linux-gnu/pkgconfig/ CC="powerpc64le-linux-gnu-gcc" CXX="powerpc64le-linux-gnu-g++" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=ON $GPU_FLAGS -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        fi
        make -j $MAKEJ
        make install/strip
        ;;
    macosx-*)
        $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=OFF $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/clang++ -DWITH_IPP=OFF ${BUILD_CONTRIB_X} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86)
        BUILD_X="-DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF"
        WITH_X="-DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=OFF $GPU_FLAGS -DWITH_IPP=OFF"
        "$CMAKE" -G "Visual Studio 14 2015" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=ON $WITH_X $BUILD_CONTRIB_X -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //maxcpucount:$MAKEJ
        cp -r ../x86/vc14/lib ..
        cp -r ../x86/vc14/bin ..
        ;;
    windows-x86_64)
        BUILD_X="-DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_hdf=OFF"
        WITH_X="-DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENMP=OFF $GPU_FLAGS -DWITH_IPP=OFF"
        "$CMAKE" -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=ON $WITH_X $BUILD_CONTRIB_X -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //maxcpucount:$MAKEJ
        cp -r ../x64/vc14/lib ..
        cp -r ../x64/vc14/bin ..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
