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
export TF_NEED_NGRAPH=0
export TF_NEED_IGNITE=0
export TF_NEED_ROCM=0
export TF_ENABLE_XLA=0
export TF_CUDA_CLANG=0
export TF_CUDA_VERSION=10.0
export TF_CUDNN_VERSION=7
export TF_DOWNLOAD_CLANG=0
export TF_NCCL_VERSION=1.3
export TF_TENSORRT_VERSION=5.0.0
export GCC_HOST_COMPILER_PATH=$(which gcc)
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export CUDNN_INSTALL_PATH=$CUDA_TOOLKIT_PATH
export TENSORRT_INSTALL_PATH=/usr/local/tensorrt/lib
export TF_CUDA_COMPUTE_CAPABILITIES=3.0
export TF_SET_ANDROID_WORKSPACE=0

TENSORFLOW_VERSION=1.12.0

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

export BUILDTARGETS="//tensorflow:libtensorflow_cc.so //tensorflow/java:tensorflow"
export BUILDFLAGS=

case $PLATFORM in
    # Clang is incapable of compiling TensorFlow for Android, while in $ANDROID_NDK/source.properties,
    # the value of Pkg.Revision needs to start with "12" for Bazel to accept GCC
    # Also, the last version of the NDK supported by TensorFlow is android-ndk-r15c
    android-arm)
        patch -Np1 < ../../../tensorflow-android.patch
        sedinplace "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        export BUILDFLAGS="--android_compiler=gcc-4.9 --crosstool_top=//external:android/crosstool --cpu=armeabi-v7a --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --copt=-DSIZE_MAX=UINT32_MAX --copt=-std=c++11 --linkopt=-s --copt=-D__user="
        ;;
    android-arm64)
        patch -Np1 < ../../../tensorflow-android.patch
        sedinplace "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        sedinplace "s/api_level=14/api_level=21/g" WORKSPACE
        export BUILDFLAGS="--android_compiler=gcc-4.9 --crosstool_top=//external:android/crosstool --cpu=arm64-v8a --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --copt=-DSIZE_MAX=UINT64_MAX --copt=-std=c++11 --linkopt=-s"
        ;;
    android-x86)
        patch -Np1 < ../../../tensorflow-android.patch
        sedinplace "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        export BUILDFLAGS="--android_compiler=gcc-4.9 --crosstool_top=//external:android/crosstool --cpu=x86 --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --copt=-DSIZE_MAX=UINT32_MAX --copt=-std=c++11 --linkopt=-s --copt=-D__user="
        ;;
    android-x86_64)
        patch -Np1 < ../../../tensorflow-android.patch
        sedinplace "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        sedinplace "s/api_level=14/api_level=21/g" WORKSPACE
        export BUILDFLAGS="--android_compiler=gcc-4.9 --crosstool_top=//external:android/crosstool --cpu=x86_64 --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --copt=-DSIZE_MAX=UINT64_MAX --copt=-std=c++11 --linkopt=-s"
        ;;
    linux-x86)
        patch -Np1 < ../../../tensorflow-java.patch
        # BoringSSL doesn't build on linux-x86, so disable secure grpc and leave undefined symbols
        patch -Np1 < ../../../tensorflow-unsecure.patch
        sedinplace '/-z defs/d' tensorflow/BUILD
        sedinplace "/        \":k8\": \[\":simd_x86_64\"\],/c\        \":k8\": \[\":simd_none\"\]," third_party/jpeg/jpeg.BUILD
        export BUILDFLAGS="--copt=-m32 --linkopt=-m32 --linkopt=-s"
        ;;
    linux-x86_64)
        patch -Np1 < ../../../tensorflow-java.patch
        export TF_NEED_MKL=1
        export BUILDFLAGS="--config=mkl --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx `#--copt=-mavx2 --copt=-mfma` $GPU_FLAGS --copt=-m64 --linkopt=-m64 --linkopt=-s"
        export CUDA_HOME=$CUDA_TOOLKIT_PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
        ;;
    macosx-*)
        # https://github.com/tensorflow/tensorflow/issues/14174
        sedinplace 's/__align__(sizeof(T))//g' tensorflow/core/kernels/*.cu.cc
        sedinplace '/-lgomp/d' third_party/gpus/cuda/BUILD.tpl
        patch -Np1 < ../../../tensorflow-java.patch
        # allows us to use ccache with Bazel
        export BAZEL_USE_CPP_ONLY_TOOLCHAIN=1
        # no longer needed? https://github.com/tensorflow/tensorflow/issues/19676
        # patch -Np1 < ../../../tensorflow-macosx.patch || true
        export TF_NEED_MKL=1
        export BUILDFLAGS="--config=mkl --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx `#--copt=-mavx2 --copt=-mfma` $GPU_FLAGS --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH --linkopt=-install_name --linkopt=@rpath/libtensorflow_cc.so --linkopt=-s"
        export CUDA_HOME=$CUDA_TOOLKIT_PATH
        export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
        export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
        export PATH=$DYLD_LIBRARY_PATH:$PATH
        ;;
    windows-x86_64)
        patch -Np1 < ../../../tensorflow-java.patch
        sedinplace 's:cuda/include/cuda_fp16.h:cuda_fp16.h:g' tensorflow/core/util/cuda_kernel_helper.h
        sedinplace 's/{diff_dst_index}, diff_src_index/{(int)diff_dst_index}, (int)diff_src_index/g' tensorflow/core/kernels/mkl_relu_op.cc
        export PYTHON_BIN_PATH=$(which python.exe)
        export BAZEL_VC="C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/"
        # try not to use /WHOLEARCHIVE as it crashes link.exe
        export NO_WHOLE_ARCHIVE_OPTION=1
        # disable __forceinline for Eigen to speed up the build
        export TF_OVERRIDE_EIGEN_STRONG_INLINE=1
        export TF_NEED_MKL=1
        export BUILDTARGETS="///tensorflow:tensorflow_static ///tensorflow/java:tensorflow"
        export BUILDFLAGS="--config=mkl --copt=/arch:AVX `#--copt=/arch:AVX2` $GPU_FLAGS --copt=/machine:x64 --linkopt=/machine:x64"
        export CUDA_TOOLKIT_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v$TF_CUDA_VERSION"
        export CUDA_HOME="$CUDA_TOOLKIT_PATH"
        export CUDNN_INSTALL_PATH="$CUDA_TOOLKIT_PATH"
# old hacks for the now obsolete CMake build
#        # help cmake's findCuda-method to find the right cuda version
#        export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v$TF_CUDA_VERSION"
#        patch -Np1 < ../../../tensorflow-java.patch
#        patch -Np1 < ../../../tensorflow-windows.patch || true
#        sedinplace 's:cuda/include/cuda_fp16.h:cuda_fp16.h:g' tensorflow/core/util/cuda_kernel_helper.h
#        sedinplace 's/mklml_win_2018.0.3.20180406.zip/mklml_win_2019.0.20180710.zip/g' tensorflow/contrib/cmake/external/mkl.cmake
#        sedinplace 's/mklml_mac_2018.0.3.20180406.tgz/mklml_mac_2019.0.20180710.tgz/g' tensorflow/contrib/cmake/external/mkl.cmake
#        sedinplace 's/mklml_lnx_2018.0.3.20180406.tgz/mklml_lnx_2019.0.20180710.tgz/g' tensorflow/contrib/cmake/external/mkl.cmake
#        sedinplace 's/v0.14/v0.16/g' tensorflow/contrib/cmake/external/mkl.cmake
#        sedinplace 's/3063b2e4c943983f6bf5f2fb9a490d4a998cd291/v0.16/g' tensorflow/contrib/cmake/external/mkldnn.cmake
#        sedinplace 's/{diff_dst_index}, diff_src_index/{(int)diff_dst_index}, (int)diff_src_index/g' tensorflow/core/kernels/mkl_relu_op.cc
#        mkdir -p ../build
#        cd ../build
#        # Disable __forceinline for Eigen to speed up the build
#        "$CMAKE" -A x64 -DCMAKE_BUILD_TYPE=Release -Dtensorflow_DISABLE_EIGEN_FORCEINLINE=ON -DPYTHON_EXECUTABLE="C:/Python27/python.exe" -DSWIG_EXECUTABLE="C:/swigwin-3.0.12/swig.exe" -Dtensorflow_BUILD_PYTHON_BINDINGS=ON -Dtensorflow_BUILD_SHARED_LIB=ON -Dtensorflow_WIN_CPU_SIMD_OPTIONS=/arch:AVX -G"Visual Studio 14" -Dtensorflow_ENABLE_MKL_SUPPORT=ON -Dtensorflow_ENABLE_MKLDNN_SUPPORT=ON $CMAKE_GPU_FLAGS -DCUDNN_HOME="$CUDA_PATH" ../tensorflow-$TENSORFLOW_VERSION/tensorflow/contrib/cmake
#        if [[ ! -f ../build/Release/tensorflow_static.lib ]]; then
#            MSBuild.exe //p:Configuration=Release //p:CL_MPCount=$MAKEJ //p:Platform=x64 //p:PreferredToolArchitecture=x64 //filelogger tensorflow_static.vcxproj
#        fi
#        if [[ ! -f ../build/tf_c_python_api.dir/Release/tf_c_python_api.lib ]]; then
#            MSBuild.exe //p:Configuration=Release //p:CL_MPCount=$MAKEJ //p:Platform=x64 //p:PreferredToolArchitecture=x64 //filelogger tf_c_python_api.vcxproj
#        fi
#        if [[ "$EXTENSION" == *gpu ]] && [[ "${PARTIAL_CPPBUILD:-}" != "1" ]]; then
#            MSBuild.exe //p:Configuration=Release //p:CL_MPCount=$MAKEJ //p:Platform=x64 //p:PreferredToolArchitecture=x64 //filelogger tf_core_gpu_kernels.vcxproj
#        fi
#        cd ../tensorflow-$TENSORFLOW_VERSION
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

bash configure
bazel build -c opt $BUILDTARGETS --config=monolithic $BUILDFLAGS --spawn_strategy=standalone --genrule_strategy=standalone --output_filter=DONT_MATCH_ANYTHING --verbose_failures

if [[ "$PLATFORM" == windows* ]]; then
    cd bazel-tensorflow-$TENSORFLOW_VERSION
    # we need /WHOLEARCHIVE for .lib files, but link.exe crashes with it, so use .obj files instead
    # (CUDA builds produce mostly files with .a and .o extensions instead of .lib and .obj)
    find -L `pwd` -iname *.obj -o -iname *.o > objs
    # remove files with main() functions or unresolved symbols in them
    sedinplace '/nasm/d' objs
    sedinplace '/js_embed/d' objs
    sedinplace '/main.o/d' objs
    sedinplace '/js_generator.o/d' objs
    sedinplace '/cc_op_gen_main.o/d' objs
    sedinplace '/gen_proto_text_functions.o/d' objs
    sedinplace '/grpc_cpp_plugin.o/d' objs
    # convert to DOS paths with short names to prevent exceeding MAX_PATH
    cygpath -d -f objs > objs.dos
    cd ..
fi

# copy/adjust Java source files and work around loader bug in NativeLibrary.java
mkdir -p ../java
cp -r tensorflow/java/src/gen/java/* ../java
cp -r tensorflow/java/src/main/java/* ../java
cp -r tensorflow/contrib/android/java/* ../java
cp -r tensorflow/contrib/lite/java/src/main/java/* ../java
cp -r bazel-genfiles/tensorflow/java/ops/src/main/java/* ../java
sedinplace '/TensorFlow.version/d' ../java/org/tensorflow/NativeLibrary.java
# remove lines that require the Android SDK to compile
sedinplace '/Trace/d' ../java/org/tensorflow/contrib/android/TensorFlowInferenceInterface.java
# add ops files we cannot get with CMake for Windows
# patch -Np1 -d ../java < ../../../tensorflow-java-ops.patch || true

cd ../..
