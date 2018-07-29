#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tensorflow
    popd
    exit
fi

export PYTHON_BIN_PATH=$(which python)
export USE_DEFAULT_PYTHON_LIB_PATH=1
export CC_OPT_FLAGS=-O3
export TF_NEED_MKL=0
export TF_NEED_VERBS=0
export TF_NEED_JEMALLOC=0
export TF_NEED_CUDA=0
export TF_NEED_AWS=0
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_KAFKA=0
export TF_NEED_S3=0
export TF_NEED_OPENCL=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_MPI=0
export TF_NEED_GDR=0
export TF_NEED_TENSORRT=0
export TF_ENABLE_XLA=0
export TF_CUDA_CLANG=0
export TF_CUDA_VERSION=9.2
export TF_CUDNN_VERSION=7
export TF_DOWNLOAD_CLANG=0
export TF_NCCL_VERSION=1.3
export TF_TENSORRT_VERSION=4.1.2
export GCC_HOST_COMPILER_PATH=$(which gcc)
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export CUDNN_INSTALL_PATH=$CUDA_TOOLKIT_PATH
export TENSORRT_INSTALL_PATH=/usr/local/tensorrt/lib
export TF_CUDA_COMPUTE_CAPABILITIES=3.0
export TF_SET_ANDROID_WORKSPACE=0

TENSORFLOW_VERSION=1.9.0

download https://github.com/tensorflow/tensorflow/archive/v$TENSORFLOW_VERSION.tar.gz tensorflow-$TENSORFLOW_VERSION.tar.gz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"

echo "Decompressing archives"
tar --totals -xzf ../tensorflow-$TENSORFLOW_VERSION.tar.gz

# Assumes Bazel is available in the path: http://bazel.io/docs/install.html
cd tensorflow-$TENSORFLOW_VERSION

# Stop complaining about possibly incompatible CUDA versions
sedinplace "s/return (cudnn == cudnn_ver) and (cudart == cuda_ver)/return True/g" configure.py

# Stop the script from annoying us with Android stuff
sed -i="" "s/return has_any_rule/return True/g" configure.py

# Allow using std::unordered_map<tensorflow::string,tensorflow::checkpoint::TensorSliceSet::SliceInfo>
sed -i="" "s/const string tag/string tag/g" tensorflow/core/util/tensor_slice_set.h

# https://github.com/tensorflow/tensorflow/issues/15389
sed -i="" "s/c2947c341c68/034b6c3e1017/g" tensorflow/workspace.bzl
sed -i="" "s/f21f8ab8a8dbcb91cd0deeade19a043f47708d0da7a4000164cdf203b4a71e34/0a8ac1e83ef9c26c0e362bd7968650b710ce54e2d883f0df84e5e45a3abe842a/g" tensorflow/workspace.bzl

export GPU_FLAGS=
export CMAKE_GPU_FLAGS=
if [[ "$EXTENSION" == *gpu ]]; then
    export TF_NEED_CUDA=1
    export TF_NEED_TENSORRT=1
    export GPU_FLAGS="--config=cuda"
    export CMAKE_GPU_FLAGS="-Dtensorflow_ENABLE_GPU=ON -Dtensorflow_CUDA_VERSION=$TF_CUDA_VERSION -Dtensorflow_CUDNN_VERSION=$TF_CUDNN_VERSION"
fi

if [[ "$TF_NEED_CUDA" == 0 ]] || [[ ! -d "$TENSORRT_INSTALL_PATH" ]]; then
    export TF_NEED_TENSORRT=0
    unset TF_TENSORRT_VERSION
    unset TENSORRT_INSTALL_PATH
fi

case $PLATFORM in
    # Clang is incapable of compiling TensorFlow for Android, while in $ANDROID_NDK/source.properties,
    # the value of Pkg.Revision needs to start with "12" for Bazel to accept GCC
    # Also, the last version of the NDK supported by TensorFlow is android-ndk-r15c
    android-arm)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        patch -Np1 < ../../../tensorflow-android.patch
        sed -i "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        export BUILDFLAGS="--android_compiler=gcc-4.9 --crosstool_top=//external:android/crosstool --cpu=armeabi-v7a --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --copt=-DSIZE_MAX=UINT32_MAX --copt=-std=c++11 --linkopt=-s"
        ;;
    android-arm64)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        patch -Np1 < ../../../tensorflow-android.patch
        sed -i "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        sed -i "s/api_level=14/api_level=21/g" WORKSPACE
        export BUILDFLAGS="--android_compiler=gcc-4.9 --crosstool_top=//external:android/crosstool --cpu=arm64-v8a --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --copt=-DSIZE_MAX=UINT64_MAX --copt=-std=c++11 --linkopt=-s"
        ;;
    android-x86)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        patch -Np1 < ../../../tensorflow-android.patch
        sed -i "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        export BUILDFLAGS="--android_compiler=gcc-4.9 --crosstool_top=//external:android/crosstool --cpu=x86 --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --copt=-DSIZE_MAX=UINT32_MAX --copt=-std=c++11 --linkopt=-s"
        ;;
    android-x86_64)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        patch -Np1 < ../../../tensorflow-android.patch
        sed -i "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        sed -i "s/api_level=14/api_level=21/g" WORKSPACE
        export BUILDFLAGS="--android_compiler=gcc-4.9 --crosstool_top=//external:android/crosstool --cpu=x86_64 --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --copt=-DSIZE_MAX=UINT64_MAX --copt=-std=c++11 --linkopt=-s"
        ;;
    linux-x86)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        patch -Np1 < ../../../tensorflow-java.patch
        # patch -Np1 < ../../../tensorflow-unsecure.patch
        sed -i "/        \":k8\": \[\":simd_x86_64\"\],/c\        \":k8\": \[\":simd_none\"\]," third_party/jpeg/jpeg.BUILD
        export BUILDFLAGS="--copt=-m32 --linkopt=-m32 --linkopt=-s"
        ;;
    linux-x86_64)
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        patch -Np1 < ../../../tensorflow-java.patch
        export GCC_HOST_COMPILER_PATH=$CC
        export BUILDFLAGS="--copt=-msse4.1 --copt=-msse4.2 --copt=-mavx `#--copt=-mavx2 --copt=-mfma` $GPU_FLAGS --copt=-m64 --linkopt=-m64 --linkopt=-s"
        export CUDA_HOME=$CUDA_TOOLKIT_PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
        ;;
    macosx-*)
        # https://github.com/tensorflow/tensorflow/issues/14174
        sed -i '' 's/__align__(sizeof(T))//g' tensorflow/core/kernels/*.cu.cc
        # https://github.com/tensorflow/tensorflow/issues/19676
        patch -R -Np1 < ../../../tensorflow-macosx-nogpu.patch || true
        patch -Np1 < ../../../tensorflow-java.patch
        export BUILDFLAGS="--copt=-msse4.1 --copt=-msse4.2 --copt=-mavx `#--copt=-mavx2 --copt=-mfma` $GPU_FLAGS --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH --linkopt=-install_name --linkopt=@rpath/libtensorflow_cc.so --linkopt=-s"
        export CUDA_HOME=$CUDA_TOOLKIT_PATH
        export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
        export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
        export PATH=$DYLD_LIBRARY_PATH:$PATH
        patch -Np1 < ../../../tensorflow-macosx.patch
        ;;
    windows-x86_64)
        # help cmake's findCuda-method to find the right cuda version
        export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v$TF_CUDA_VERSION"
        patch -Np1 < ../../../tensorflow-java.patch
        mkdir -p ../build
        cd ../build
        "$CMAKE" -A x64 -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="C:/Python27/python.exe" -Dtensorflow_BUILD_PYTHON_BINDINGS=OFF -Dtensorflow_BUILD_SHARED_LIB=ON -Dtensorflow_WIN_CPU_SIMD_OPTIONS=/arch:AVX -G"Visual Studio 14" $CMAKE_GPU_FLAGS -DCUDNN_HOME="$CUDA_PATH" ../tensorflow-$TENSORFLOW_VERSION/tensorflow/contrib/cmake
        if [[ ! -f ../build/Release/tensorflow_static.lib ]]; then
            MSBuild.exe //p:Configuration=Release /maxcpucount:$MAKEJ tensorflow_static.vcxproj
        fi
        if [[ "$EXTENSION" == *gpu ]] && [[ "${PARTIAL_CPPBUILD:-}" != "1" ]]; then
            MSBuild.exe //p:Configuration=Release /maxcpucount:$MAKEJ tf_core_gpu_kernels.vcxproj
        fi
        cd ../tensorflow-$TENSORFLOW_VERSION
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

if [[ ! "$PLATFORM" == windows* ]]; then
    ./configure
    bazel build -c opt //tensorflow:libtensorflow_cc.so --config=monolithic $BUILDFLAGS --spawn_strategy=standalone --genrule_strategy=standalone --output_filter=DONT_MATCH_ANYTHING --verbose_failures
fi

# copy/adjust Java source files and work around loader bug in NativeLibrary.java
mkdir -p ../java
cp -r tensorflow/java/src/gen/java/* ../java
cp -r tensorflow/java/src/main/java/* ../java
cp -r tensorflow/contrib/android/java/* ../java
cp -r tensorflow/contrib/lite/java/src/main/java/* ../java
sedinplace '/TensorFlow.version/d' ../java/org/tensorflow/NativeLibrary.java
sedinplace '/Trace/d' ../java/org/tensorflow/contrib/android/TensorFlowInferenceInterface.java

cd ../..
