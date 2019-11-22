#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" opencv
    popd
    exit
fi

OPENCV_VERSION=4.1.2
download https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz opencv-$OPENCV_VERSION.tar.gz
download https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.tar.gz opencv_contrib-$OPENCV_VERSION.tar.gz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`

CPYTHON_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/"
OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"
NUMPY_PATH="$INSTALL_PATH/../../../numpy/cppbuild/$PLATFORM/"

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ $(find "$P" -name Python.h) ]]; then
            CPYTHON_PATH="$P"
        elif [[ -f "$P/include/openblas_config.h" ]]; then
            OPENBLAS_PATH="$P"
        elif [[ -f "$P/python/numpy/core/include/numpy/numpyconfig.h" ]]; then
            NUMPY_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

export OpenBLAS_HOME=$OPENBLAS_PATH

CPYTHON_PATH="${CPYTHON_PATH//\\//}"
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"
NUMPY_PATH="${NUMPY_PATH//\\//}"

export PYTHON3_EXECUTABLE=
export PYTHON3_INCLUDE_DIR=
export PYTHON3_LIBRARY=
export PYTHON3_PACKAGES_PATH=
if [[ -f "$CPYTHON_PATH/include/python3.7m/Python.h" ]]; then
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/:$NUMPY_PATH/lib/:${LD_LIBRARY_PATH:-}"
    export PYTHON3_EXECUTABLE="$CPYTHON_PATH/bin/python3.7"
    export PYTHON3_INCLUDE_DIR="$CPYTHON_PATH/include/python3.7m/"
    export PYTHON3_LIBRARY="$CPYTHON_PATH/lib/python3.7/"
    export PYTHON3_PACKAGES_PATH="$INSTALL_PATH/lib/python3.7/site-packages/"
    chmod +x "$PYTHON3_EXECUTABLE"
elif [[ -f "$CPYTHON_PATH/include/Python.h" ]]; then
    CPYTHON_PATH=$(cygpath $CPYTHON_PATH)
    OPENBLAS_PATH=$(cygpath $OPENBLAS_PATH)
    NUMPY_PATH=$(cygpath $NUMPY_PATH)
    export PATH="$OPENBLAS_PATH:$CPYTHON_PATH:$NUMPY_PATH:$PATH"
    export PYTHON3_EXECUTABLE="$CPYTHON_PATH/bin/python.exe"
    export PYTHON3_INCLUDE_DIR="$CPYTHON_PATH/include/"
    export PYTHON3_LIBRARY="$CPYTHON_PATH/libs/python37.lib"
    export PYTHON3_PACKAGES_PATH="$INSTALL_PATH/lib/site-packages/"
fi
export PYTHONPATH="$NUMPY_PATH/python/:${PYTHONPATH:-}"

echo "Decompressing archives..."
tar --totals -xzf ../opencv-$OPENCV_VERSION.tar.gz
tar --totals -xzf ../opencv_contrib-$OPENCV_VERSION.tar.gz

cd opencv_contrib-$OPENCV_VERSION
patch -Np1 < ../../../opencv_contrib.patch

cd ../opencv-$OPENCV_VERSION
patch -Np1 < ../../../opencv.patch
patch -Np1 < ../../../opencv-linux-ppc64le.patch

# work around the toolchain for Android not supporting Clang with libc++ properly
sedinplace '/include_directories/d' platforms/android/android.toolchain.cmake
sedinplace "s/<LINK_LIBRARIES>/<LINK_LIBRARIES> ${ANDROID_LIBS:-}/g" platforms/android/android.toolchain.cmake

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

# fixes for CUDA
sedinplace '/typedef ::/d' modules/core/include/opencv2/core/cvdef.h
sedinplace 's/__constant__//g' modules/core/include/opencv2/core/cuda/detail/color_detail.hpp

# avoid issues when checking version of cross-compiled Python
sedinplace '/if(PYTHONINTERP_FOUND)/a\
    if(" ${_python_version_major}" STREQUAL " 3")\
      set(PYTHON_VERSION_STRING "3.7")\
      set(PYTHON_VERSION_MAJOR "3")\
      set(PYTHON_VERSION_MINOR "7")\
    endif()\
' cmake/OpenCVDetectPython.cmake
sedinplace '/execute_process/{N;N;N;d;}' cmake/OpenCVDetectPython.cmake

BUILD_X="-DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_ANDROID_PROJECTS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_opencv_java=ON -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON -DPYTHON3_EXECUTABLE=$PYTHON3_EXECUTABLE -DPYTHON3_INCLUDE_DIR=$PYTHON3_INCLUDE_DIR -DPYTHON3_LIBRARY=$PYTHON3_LIBRARY -DPYTHON3_PACKAGES_PATH=$PYTHON3_PACKAGES_PATH -DPYTHON3_NUMPY_INCLUDE_DIRS=$NUMPY_PATH/python/numpy/core/include/ -DBUILD_opencv_gapi=OFF -DBUILD_opencv_hdf=OFF -DBUILD_opencv_img_hash=ON"

# support for OpenMP is NOT thread-safe so make sure to never enable it and use pthreads instead
WITH_X="-DWITH_1394=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_IPP=OFF -DWITH_LAPACK=ON -DWITH_OPENCL=ON -DWITH_OPENMP=OFF -DOPENCV_ENABLE_NONFREE=ON"

# support headless
if [[ "${HEADLESS:-no}" == "yes" ]]; then
    WITH_X="$WITH_X -DWITH_WIN32UI=OFF -DWITH_QT=OFF -DWITH_GTK=OFF"
fi

BUILD_CONTRIB_X="-DBUILD_opencv_stereo=OFF -DBUILD_opencv_plot=ON -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_aruco=ON -DBUILD_opencv_adas=OFF -DBUILD_opencv_bgsegm=ON -DBUILD_opencv_bioinspired=ON -DBUILD_opencv_ccalib=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn_modern=OFF -DBUILD_opencv_dnn_objdetect=OFF -DBUILD_opencv_freetype=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=ON -DBUILD_opencv_hfs=OFF -DBUILD_opencv_latentsvm=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_matlab=OFF -DBUILD_opencv_optflow=ON -DBUILD_opencv_phase_unwrapping=ON -DBUILD_opencv_plot=ON -DBUILD_opencv_reg=OFF -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=ON -DBUILD_opencv_structured_light=ON -DBUILD_opencv_surface_matching=OFF -DBUILD_opencv_text=ON -DBUILD_opencv_tracking=ON -DBUILD_opencv_xfeatures2d=ON -DBUILD_opencv_ximgproc=ON -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules"

GPU_FLAGS="-DWITH_CUDA=OFF"
if [[ "$EXTENSION" == *gpu ]]; then
    GPU_FLAGS="-DWITH_CUDA=ON -DCUDA_VERSION=10.1 -DCUDA_ARCH_BIN=3.0 -DCUDA_ARCH_PTX=3.0 -DCUDA_NVCC_FLAGS=--expt-relaxed-constexpr"
fi

case $PLATFORM in
    android-arm)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9 -DANDROID_STL=c++_static -DANDROID_NATIVE_API_LEVEL=21 -DCMAKE_C_FLAGS="$ANDROID_FLAGS" -DCMAKE_CXX_FLAGS="$ANDROID_FLAGS" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $GPU_FLAGS $BUILD_CONTRIB_X .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/armeabi-v7a/* ../lib
        sedinplace 's:/sdk/native/jni/include:/include:g' ../sdk/native/jni/abi-armeabi-v7a/OpenCVConfig.cmake
        sedinplace 's:/sdk/native/libs/armeabi-v7a/libopencv_:/lib/libopencv_:g' ../sdk/native/jni/abi-armeabi-v7a/OpenCVModules-release.cmake
        ;;
    android-arm64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-4.9 -DANDROID_STL=c++_static -DANDROID_NATIVE_API_LEVEL=21 -DCMAKE_C_FLAGS="$ANDROID_FLAGS" -DCMAKE_CXX_FLAGS="$ANDROID_FLAGS" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $GPU_FLAGS $BUILD_CONTRIB_X .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/arm64-v8a/* ../lib
        sedinplace 's:/sdk/native/jni/include:/include:g' ../sdk/native/jni/abi-arm64-v8a/OpenCVConfig.cmake
        sedinplace 's:/sdk/native/libs/arm64-v8a/libopencv_:/lib/libopencv_:g' ../sdk/native/jni/abi-arm64-v8a/OpenCVModules-release.cmake
        ;;
    android-x86)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=x86 -DANDROID_TOOLCHAIN_NAME=x86-4.9 -DANDROID_STL=c++_static -DANDROID_NATIVE_API_LEVEL=21 -DCMAKE_C_FLAGS="$ANDROID_FLAGS" -DCMAKE_CXX_FLAGS="$ANDROID_FLAGS" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $GPU_FLAGS $BUILD_CONTRIB_X .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/x86/* ../lib
        sedinplace 's:/sdk/native/jni/include:/include:g' ../sdk/native/jni/abi-x86/OpenCVConfig.cmake
        sedinplace 's:/sdk/native/libs/x86/libopencv_:/lib/libopencv_:g' ../sdk/native/jni/abi-x86/OpenCVModules-release.cmake
        ;;
    android-x86_64)
        $CMAKE -DCMAKE_TOOLCHAIN_FILE=platforms/android/android.toolchain.cmake -DANDROID_ABI=x86_64 -DANDROID_TOOLCHAIN_NAME=x86_64-4.9 -DANDROID_STL=c++_static -DANDROID_NATIVE_API_LEVEL=21 -DCMAKE_C_FLAGS="$ANDROID_FLAGS" -DCMAKE_CXX_FLAGS="$ANDROID_FLAGS" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DBUILD_SHARED_LIBS=ON $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $GPU_FLAGS $BUILD_CONTRIB_X .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install/strip
        cp -r ../sdk/native/jni/include ..
        mkdir -p ../lib
        cp ../sdk/native/libs/x86_64/* ../lib
        sedinplace 's:/sdk/native/jni/include:/include:g' ../sdk/native/jni/abi-x86_64/OpenCVConfig.cmake
        sedinplace 's:/sdk/native/libs/x86_64/libopencv_:/lib/libopencv_:g' ../sdk/native/jni/abi-x86_64/OpenCVModules-release.cmake
        ;;
    ios-arm64)
        $CMAKE -GXcode -DCMAKE_TOOLCHAIN_FILE=platforms/ios/cmake/Toolchains/Toolchain-iPhoneOS_Xcode.cmake -DIPHONEOS_DEPLOYMENT_TARGET=8.0 -DIOS_ARCH=arm64 -DAPPLE_FRAMEWORK=ON -DCMAKE_MACOSX_BUNDLE=ON -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO -DCMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE=NO -DBUILD_opencv_world=OFF -DBUILD_SHARED_LIBS=OFF $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENCL=OFF $GPU_FLAGS $BUILD_CONTRIB_X -DCMAKE_CXX_FLAGS="-I/System/Library/Frameworks/JavaVM.framework/Versions/A/Headers/" .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        xcodebuild -arch arm64 -sdk iphoneos -configuration Release -parallelizeTargets -jobs $MAKEJ ONLY_ACTIVE_ARCH=NO -target ALL_BUILD build > /dev/null
        $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -P cmake_install.cmake .
        cp ../share/java/opencv4/libopencv_java.a ../lib
        ;;
    ios-x86_64)
        $CMAKE -GXcode -DCMAKE_TOOLCHAIN_FILE=platforms/ios/cmake/Toolchains/Toolchain-iPhoneSimulator_Xcode.cmake -DIPHONEOS_DEPLOYMENT_TARGET=8.0 -DIOS_ARCH=x86_64 -DAPPLE_FRAMEWORK=ON -DCMAKE_MACOSX_BUNDLE=ON -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO -DCMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE=NO -DBUILD_opencv_world=OFF -DBUILD_SHARED_LIBS=OFF $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_OPENCL=OFF $GPU_FLAGS $BUILD_CONTRIB_X -DCMAKE_CXX_FLAGS="-I/System/Library/Frameworks/JavaVM.framework/Versions/A/Headers/" .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        xcodebuild -arch x86_64 -sdk iphonesimulator -configuration Release -parallelizeTargets -jobs $MAKEJ ONLY_ACTIVE_ARCH=NO -target ALL_BUILD build > /dev/null
        $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -P cmake_install.cmake .
        cp ../share/java/opencv4/libopencv_java.a ../lib
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -std=c++11 -m32" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DENABLE_SSE3=OFF $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $GPU_FLAGS -DCUDA_HOST_COMPILER="$(which g++)" $BUILD_CONTRIB_X -DCMAKE_CXX_FLAGS="-w" .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install/strip
        cp ../share/java/opencv4/libopencv_java.so ../lib
        sedinplace "s/.so.${OPENCV_VERSION%-*}/.so/g" ../lib/cmake/opencv4/OpenCVModules-release.cmake
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -std=c++11 -m64" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $GPU_FLAGS -DCUDA_HOST_COMPILER="$(which g++)" $BUILD_CONTRIB_X -DCMAKE_CXX_FLAGS="-w" .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install/strip
        cp ../share/java/opencv4/libopencv_java.so ../lib
        sedinplace "s/.so.${OPENCV_VERSION%-*}/.so/g" ../lib/cmake/opencv4/OpenCVModules-release.cmake
        ;;
    linux-arm)
        CC="arm-linux-gnueabi-gcc" CXX="arm-linux-gnueabi-g++ -std=c++11" CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_GTK=OFF $GPU_FLAGS -DCUDA_HOST_COMPILER="$(which arm-linux-gnueabi-g++)" $BUILD_CONTRIB_X .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install
        cp ../share/java/opencv4/libopencv_java.so ../lib
        sedinplace "s/.so.${OPENCV_VERSION%-*}/.so/g" ../lib/cmake/opencv4/OpenCVModules-release.cmake
        ;;
    linux-armhf)
        CC="arm-linux-gnueabihf-gcc" CXX="arm-linux-gnueabihf-g++ -std=c++11" CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DCMAKE_SYSTEM_PROCESSOR=armv6 -DBUILD_TESTS=OFF -DCMAKE_CXX_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard -Wl,-allow-shlib-undefined" -DCMAKE_C_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard -Wl,-allow-shlib-undefined" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_GTK=OFF $GPU_FLAGS -DCUDA_HOST_COMPILER="$(which arm-linux-gnueabihf-g++)" $BUILD_CONTRIB_X .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install/strip
        cp ../share/java/opencv4/libopencv_java.so ../lib
        sedinplace "s/.so.${OPENCV_VERSION%-*}/.so/g" ../lib/cmake/opencv4/OpenCVModules-release.cmake
        ;;
    linux-arm64)
        PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig/ CC="aarch64-linux-gnu-gcc -DITT_ARCH=4 -I/usr/lib/jvm/default-java/include/ -I/usr/lib/jvm/default-java/include/linux/" CXX="aarch64-linux-gnu-g++ -std=c++11 -DITT_ARCH=4 -I/usr/lib/jvm/default-java/include/ -I/usr/lib/jvm/default-java/include/linux/" CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DAARCH64=ON -DENABLE_NEON=OFF -DENABLE_SSE=OFF -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DBUILD_TESTS=OFF -DCMAKE_CXX_FLAGS="" -DCMAKE_C_FLAGS="" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X -DWITH_GTK=OFF $GPU_FLAGS -DCUDA_HOST_COMPILER="$(which aarch64-linux-gnu-g++)" $BUILD_CONTRIB_X .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install/strip
        cp ../share/java/opencv4/libopencv_java.so ../lib
        sedinplace "s/.so.${OPENCV_VERSION%-*}/.so/g" ../lib/cmake/opencv4/OpenCVModules-release.cmake
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64 -mpower8-vector -DCV_CPU_COMPILE_VSX -I/usr/lib/jvm/default-java/include/" CXX="g++ -std=c++11 -m64 -mpower8-vector -DCV_CPU_COMPILE_VSX -I/usr/lib/jvm/default-java/include/" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $GPU_FLAGS -DCUDA_HOST_COMPILER="$(which g++)" $BUILD_CONTRIB_X .
        else
          echo "Not native ppc so assume cross compiling"
          PKG_CONFIG_PATH=/usr/lib/powerpc64le-linux-gnu/pkgconfig/ CC="powerpc64le-linux-gnu-gcc -m64 -mpower8-vector -DCV_CPU_COMPILE_VSX -I/usr/lib/jvm/default-java/include/ -I/usr/lib/jvm/default-java/include/linux/" CXX="powerpc64le-linux-gnu-g++ -std=c++11 -m64 -mpower8-vector -DCV_CPU_COMPILE_VSX -I/usr/lib/jvm/default-java/include/ -I/usr/lib/jvm/default-java/include/linux/" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $GPU_FLAGS $BUILD_CONTRIB_X .
        fi
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install/strip
        cp ../share/java/opencv4/libopencv_java.so ../lib
        sedinplace "s/.so.${OPENCV_VERSION%-*}/.so/g" ../lib/cmake/opencv4/OpenCVModules-release.cmake
        ;;
    linux-mips64el)
        CC="gcc -mabi=64" CXX="g++ -std=c++11 -mabi=64" $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $GPU_FLAGS -DCUDA_HOST_COMPILER="$(which g++)" $BUILD_CONTRIB_X .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install/strip
        cp ../share/java/opencv4/libopencv_java.so ../lib
        sedinplace "s/.so.${OPENCV_VERSION%-*}/.so/g" ../lib/cmake/opencv4/OpenCVModules-release.cmake
        ;;
    macosx-*)
        # also use pthreads on Mac for increased usability and more consistent behavior with Linux
        sedinplace '/HAVE_GCD/d' CMakeLists.txt
        # remove spurious "lib" lib
        sedinplace '/if.*(HAVE_CUDA)/a\
            list(REMOVE_ITEM CUDA_LIBRARIES lib)\
            ' CMakeLists.txt cmake/OpenCVModule.cmake cmake/OpenCVDetectCUDA.cmake
        $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=OFF $WITH_X $GPU_FLAGS -DCUDA_HOST_COMPILER=/usr/bin/clang++ $BUILD_CONTRIB_X -DCMAKE_CXX_FLAGS="-w" .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        make -j $MAKEJ
        make install/strip
        cp ../share/java/opencv4/libopencv_java.dylib ../lib
        sedinplace "s/.${OPENCV_VERSION%-*}.dylib/.dylib/g" ../lib/cmake/opencv4/OpenCVModules-release.cmake
        ;;
    windows-x86)
        "$CMAKE" -G "Visual Studio 15 2017" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=ON $WITH_X $GPU_FLAGS $BUILD_CONTRIB_X -DPYTHON_EXECUTABLE="C:/Python27/python.exe" .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        cp -r ../x86/vc15/lib ..
        cp -r ../x86/vc15/bin ..
        cp lib/Release/opencv_java.lib ../lib
        cp lib/Release/opencv_java.dll ../bin
        sedinplace "s:/x86/vc15/lib/:/lib/:g" ../x86/vc15/lib/OpenCVModules-release.cmake
        sedinplace "s:/x86/vc15/bin/:/:g" ../x86/vc15/lib/OpenCVModules-release.cmake
        ;;
    windows-x86_64)
        "$CMAKE" -G "Visual Studio 15 2017 Win64" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_INSTALL_LIBDIR="lib" $BUILD_X -DENABLE_PRECOMPILED_HEADERS=ON $WITH_X $GPU_FLAGS $BUILD_CONTRIB_X -DPYTHON_EXECUTABLE="C:/Python27/python.exe" .
        # download files CMake failed to download
        if [[ -f download_with_curl.sh ]]; then
            bash download_with_curl.sh
            $CMAKE .
        fi
        # work around some bug in the CUDA build
        [[ ! -f modules/cudev/opencv_cudev_main.cpp ]] || sedinplace '/__termination/d' modules/cudev/opencv_cudev_main.cpp
        MSBuild.exe INSTALL.vcxproj //p:Configuration=Release //p:CL_MPCount=$MAKEJ
        cp -r ../x64/vc15/lib ..
        cp -r ../x64/vc15/bin ..
        cp lib/Release/opencv_java.lib ../lib
        cp lib/Release/opencv_java.dll ../bin
        sedinplace "s:/x64/vc15/lib/:/lib/:g" ../x64/vc15/lib/OpenCVModules-release.cmake
        sedinplace "s:/x64/vc15/bin/:/:g" ../x64/vc15/lib/OpenCVModules-release.cmake
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

# link the include directory back to a sane default
if [[ -d ../include/opencv4 ]]; then
    ln -sf opencv4/opencv2 ../include
fi

cp -r modules/java_bindings_generator/gen/java ..
cp -r modules/java_bindings_generator/gen/android/java ..
# remove files that require the Android SDK to compile
rm ../java/org/opencv/android/AsyncServiceHelper.java
rm ../java/org/opencv/android/CameraActivity.java
rm ../java/org/opencv/android/CameraBridgeViewBase.java
rm ../java/org/opencv/android/JavaCameraView.java
rm ../java/org/opencv/android/OpenCVLoader.java
rm ../java/*opencv* || true # remove stray binaries

if [[ -d ${PYTHON3_PACKAGES_PATH:-} ]]; then
    ln -snf ${PYTHON3_PACKAGES_PATH:-} ../python
    rm -Rf $(find ../ -iname __pycache__)
fi

cd ../..
