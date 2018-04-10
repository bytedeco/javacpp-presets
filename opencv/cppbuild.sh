#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" opencv
    popd
    exit
fi

OPENCV_VERSION=3.4.1
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
patch -Np1 < ../../../opencv_java.patch

# fixes for iOS
if [[ $PLATFORM == ios* ]]; then
    sed -i="" '/#if defined(NO)/a\
    #undef NO\
    ' modules/stitching/include/opencv2/stitching/detail/exposure_compensate.hpp

    sed -i="" '/project(libprotobuf)/a\
    add_definitions(-O1)\
    ' 3rdparty/protobuf/CMakeLists.txt

    sed -i="" '/add_definitions(-DHAVE_PROTOBUF=1)/a\
    add_definitions(-O1)\
    ' modules/dnn/CMakeLists.txt
fi

# fix for CUDA
sed -i="" '/typedef ::/d' modules/core/include/opencv2/core/cvdef.h

BUILD_X="-DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=ON -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_hdf=OFF -DBUILD_opencv_img_hash=ON"

WITH_X="-DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_LAPACK=OFF -DWITH_OPENCL=ON"

BUILD_CONTRIB_X="-DBUILD_opencv_stereo=OFF -DBUILD_opencv_plot=ON -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_aruco=ON -DBUILD_opencv_adas=OFF -DBUILD_opencv_bgsegm=ON -DBUILD_opencv_bioinspired=ON -DBUILD_opencv_ccalib=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn_modern=OFF -DBUILD_opencv_dnn_objdetect=OFF -DBUILD_opencv_freetype=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=ON -DBUILD_opencv_hfs=OFF -DBUILD_opencv_latentsvm=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_matlab=OFF -DBUILD_opencv_optflow=ON -DBUILD_opencv_phase_unwrapping=ON -DBUILD_opencv_plot=ON -DBUILD_opencv_reg=OFF -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_structured_light=ON -DBUILD_opencv_surface_matching=OFF -DBUILD_opencv_text=ON -DBUILD_opencv_tracking=ON -DBUILD_opencv_xfeatures2d=ON -DBUILD_opencv_ximgproc=ON -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules"

GPU_FLAGS="-DWITH_CUDA=OFF"
if [[ "$EXTENSION" == *gpu ]]; then
    GPU_FLAGS="-DWITH_CUDA=ON -DCUDA_VERSION=9.1 -DCUDA_ARCH_BIN=3.0 -DCUDA_ARCH_PTX=3.0 -DCUDA_NVCC_FLAGS=--expt-relaxed-constexpr"
fi

case $PLATFORM in
    android-arm)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9 -DANDROID_NATIVE_API_LEVEL=14 -DHAVE_MMAP=0 -DCMAKE_C_FLAGS="$ANDROID_FLAGS" -DCMAKE_CXX_FLAGS="$ANDROID_FLAGS" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENMP=ON $GPU_FLAGS -DWITH_IPP=OFF $BUILD_CONTRIB_X
        make -j $MAKEJ
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/armeabi-v7a/* ../lib
        sedinplace 's:/sdk/native/jni/include:/include:g' ../sdk/native/jni/abi-armeabi-v7a/OpenCVConfig.cmake
        sedinplace 's:/sdk/native/libs/armeabi-v7a/libopencv_:/lib/libopencv_:g' ../sdk/native/jni/abi-armeabi-v7a/OpenCVModules-release.cmake
        ;;
    android-arm64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-4.9 -DANDROID_NATIVE_API_LEVEL=21 -DCMAKE_C_FLAGS="$ANDROID_FLAGS" -DCMAKE_CXX_FLAGS="$ANDROID_FLAGS" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENMP=ON $GPU_FLAGS -DWITH_IPP=OFF $BUILD_CONTRIB_X
        make -j $MAKEJ
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/arm64-v8a/* ../lib
        sedinplace 's:/sdk/native/jni/include:/include:g' ../sdk/native/jni/abi-arm64-v8a/OpenCVConfig.cmake
        sedinplace 's:/sdk/native/libs/arm64-v8a/libopencv_:/lib/libopencv_:g' ../sdk/native/jni/abi-arm64-v8a/OpenCVModules-release.cmake
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_TOOLCHAIN_NAME=x86-4.9 -DANDROID_NATIVE_API_LEVEL=14 -DHAVE_MMAP=0 -DCMAKE_C_FLAGS="$ANDROID_FLAGS" -DCMAKE_CXX_FLAGS="$ANDROID_FLAGS" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENMP=ON $GPU_FLAGS -DWITH_IPP=OFF $BUILD_CONTRIB_X
        make -j $MAKEJ
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/x86/* ../lib
        sedinplace 's:/sdk/native/jni/include:/include:g' ../sdk/native/jni/abi-x86/OpenCVConfig.cmake
        sedinplace 's:/sdk/native/libs/x86/libopencv_:/lib/libopencv_:g' ../sdk/native/jni/abi-x86/OpenCVModules-release.cmake
        ;;
    android-x86_64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_TOOLCHAIN_NAME=x86_64-4.9 -DANDROID_NATIVE_API_LEVEL=21 -DCMAKE_C_FLAGS="$ANDROID_FLAGS" -DCMAKE_CXX_FLAGS="$ANDROID_FLAGS" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENMP=ON $GPU_FLAGS -DWITH_IPP=OFF $BUILD_CONTRIB_X
        make -j $MAKEJ
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/x86_64/* ../lib
        sedinplace 's:/sdk/native/jni/include:/include:g' ../sdk/native/jni/abi-x86_64/OpenCVConfig.cmake
        sedinplace 's:/sdk/native/libs/x86_64/libopencv_:/lib/libopencv_:g' ../sdk/native/jni/abi-x86_64/OpenCVModules-release.cmake
        ;;
    ios-arm64)
        $CMAKE -GXcode -DCMAKE_TOOLCHAIN_FILE=platforms/ios/cmake/Toolchains/Toolchain-iPhoneOS_Xcode.cmake -DIOS_ARCH=arm64 -DAPPLE_FRAMEWORK=ON -DCMAKE_MACOSX_BUNDLE=ON -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO -DCMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE=NO -DBUILD_opencv_world=OFF -DBUILD_SHARED_LIBS=OFF $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENMP=OFF $GPU_FLAGS -DWITH_IPP=OFF $BUILD_CONTRIB_X -DCMAKE_CXX_FLAGS="-I/System/Library/Frameworks/JavaVM.framework/Versions/A/Headers/"
        xcodebuild -arch arm64 -sdk iphoneos -configuration Release -parallelizeTargets -jobs $MAKEJ ONLY_ACTIVE_ARCH=NO -target ALL_BUILD build
        $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -P cmake_install.cmake
        cp ../share/OpenCV/3rdparty/lib/* ../lib
        cp ../share/OpenCV/java/libopencv_java.a ../lib
        ;;
    ios-x86_64)
        $CMAKE -GXcode -DCMAKE_TOOLCHAIN_FILE=platforms/ios/cmake/Toolchains/Toolchain-iPhoneSimulator_Xcode.cmake -DIOS_ARCH=x86_64 -DAPPLE_FRAMEWORK=ON -DCMAKE_MACOSX_BUNDLE=ON -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO -DCMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE=NO -DBUILD_opencv_world=OFF -DBUILD_SHARED_LIBS=OFF $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENMP=OFF $GPU_FLAGS -DWITH_IPP=OFF $BUILD_CONTRIB_X -DCMAKE_CXX_FLAGS="-I/System/Library/Frameworks/JavaVM.framework/Versions/A/Headers/"
        xcodebuild -arch x86_64 -sdk iphonesimulator -configuration Release -parallelizeTargets -jobs $MAKEJ ONLY_ACTIVE_ARCH=NO -target ALL_BUILD build
        $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -P cmake_install.cmake
        cp ../share/OpenCV/3rdparty/lib/* ../lib
        cp ../share/OpenCV/java/libopencv_java.a ../lib
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DENABLE_SSE3=OFF $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENMP=ON $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF $BUILD_CONTRIB_X -DCMAKE_CXX_FLAGS="-w"
        make -j $MAKEJ
        make install/strip
        cp ../share/OpenCV/java/libopencv_java.so ../lib
        sedinplace "s/.so.$OPENCV_VERSION/.so/g" ../share/OpenCV/OpenCVModules-release.cmake
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENMP=ON $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF $BUILD_CONTRIB_X -DCMAKE_CXX_FLAGS="-w"
        make -j $MAKEJ
        make install/strip
        cp ../share/OpenCV/java/libopencv_java.so ../lib
        sedinplace "s/.so.$OPENCV_VERSION/.so/g" ../share/OpenCV/OpenCVModules-release.cmake
        ;;
    linux-arm)
        CC="arm-linux-gnueabi-gcc" CXX="arm-linux-gnueabi-g++" CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DBUILD_opencv_python3=OFF -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_GTK=OFF -DWITH_OPENMP=OFF $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF $BUILD_CONTRIB_X
        make -j $MAKEJ
        make install
        cp ../share/OpenCV/java/libopencv_java.so ../lib
        sedinplace "s/.so.$OPENCV_VERSION/.so/g" ../share/OpenCV/OpenCVModules-release.cmake
        ;;
    linux-armhf)
        CC="arm-linux-gnueabihf-gcc" CXX="arm-linux-gnueabihf-g++" CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DCMAKE_SYSTEM_PROCESSOR=armv6 -DBUILD_TESTS=OFF -DCMAKE_CXX_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard" -DCMAKE_C_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard" $BUILD_X -DBUILD_opencv_python3=OFF -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_GTK=OFF -DWITH_OPENMP=ON $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF $BUILD_CONTRIB_X
        make -j $MAKEJ
        make install/strip
        cp ../share/OpenCV/java/libopencv_java.so ../lib
        sedinplace "s/.so.$OPENCV_VERSION/.so/g" ../share/OpenCV/OpenCVModules-release.cmake
        ;;
    linux-arm64)
        CC="aarch64-linux-gnu-gcc" CXX="aarch64-linux-gnu-g++" CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DAARCH64=ON -DENABLE_NEON=OFF -DENABLE_SSE=OFF -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DBUILD_TESTS=OFF -DCMAKE_CXX_FLAGS="" -DCMAKE_C_FLAGS="" $BUILD_X -DBUILD_opencv_python3=OFF -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_GTK=OFF -DWITH_OPENMP=ON $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF $BUILD_CONTRIB_X
        make -j $MAKEJ
        make install/strip
        cp ../share/OpenCV/java/libopencv_java.so ../lib
        sedinplace "s/.so.$OPENCV_VERSION/.so/g" ../share/OpenCV/OpenCVModules-release.cmake
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENMP=ON $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/g++ -DWITH_IPP=OFF $BUILD_CONTRIB_X
        else
          echo "Not native ppc so assume cross compiling"
          PKG_CONFIG_PATH=/usr/lib/powerpc64le-linux-gnu/pkgconfig/ CC="powerpc64le-linux-gnu-gcc" CXX="powerpc64le-linux-gnu-g++" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENMP=ON $GPU_FLAGS -DWITH_IPP=OFF $BUILD_CONTRIB_X
        fi
        make -j $MAKEJ
        make install/strip
        cp ../share/OpenCV/java/libopencv_java.so ../lib
        sedinplace "s/.so.$OPENCV_VERSION/.so/g" ../share/OpenCV/OpenCVModules-release.cmake
        ;;
    macosx-*)
        $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENMP=OFF $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/clang++ -DWITH_IPP=OFF $BUILD_CONTRIB_X -DCMAKE_CXX_FLAGS="-w"
        make -j $MAKEJ
        make install/strip
        cp ../share/OpenCV/java/libopencv_java.dylib ../lib
        sedinplace "s/.$OPENCV_VERSION.dylib/.dylib/g" ../share/OpenCV/OpenCVModules-release.cmake
        ;;
    windows-x86)
        "$CMAKE" -G "Visual Studio 14 2015" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=ON $WITH_X -DWITH_OPENMP=OFF $GPU_FLAGS -DWITH_IPP=OFF $BUILD_CONTRIB_X
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //maxcpucount:$MAKEJ
        cp -r ../x86/vc14/lib ..
        cp -r ../x86/vc14/bin ..
        cp lib/Release/opencv_java.lib ../lib
        cp lib/Release/opencv_java.dll ../bin
        sedinplace "s:/x86/vc14/lib/:/lib/:g" ../x86/vc14/lib/OpenCVModules-release.cmake
        sedinplace "s:/x86/vc14/bin/:/:g" ../x86/vc14/lib/OpenCVModules-release.cmake
        ;;
    windows-x86_64)
        "$CMAKE" -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=ON $WITH_X -DWITH_OPENMP=OFF $GPU_FLAGS -DWITH_IPP=OFF $BUILD_CONTRIB_X
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //maxcpucount:$MAKEJ
        cp -r ../x64/vc14/lib ..
        cp -r ../x64/vc14/bin ..
        cp lib/Release/opencv_java.lib ../lib
        cp lib/Release/opencv_java.dll ../bin
        sedinplace "s:/x64/vc14/lib/:/lib/:g" ../x64/vc14/lib/OpenCVModules-release.cmake
        sedinplace "s:/x64/vc14/bin/:/:g" ../x64/vc14/lib/OpenCVModules-release.cmake
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cp -r modules/java_bindings_generator/gen/java ..
cp -r modules/java_bindings_generator/gen/android/java ..
rm ../java/org/opencv/android/AsyncServiceHelper.java
rm ../java/org/opencv/android/CameraBridgeViewBase.java
rm ../java/org/opencv/android/JavaCameraView.java
rm ../java/org/opencv/android/OpenCVLoader.java

cd ../..
