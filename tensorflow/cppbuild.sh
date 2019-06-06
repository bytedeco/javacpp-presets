#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tensorflow
    popd
    exit
fi

export PYTHON_BIN_PATH=$(which python3)
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
export TF_CUDA_VERSION=10.1
export TF_CUDNN_VERSION=7
export TF_DOWNLOAD_CLANG=0
export TF_NCCL_VERSION=2.4
export TF_TENSORRT_VERSION=5.1
export GCC_HOST_COMPILER_PATH=$(which gcc)
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export CUDNN_INSTALL_PATH=$CUDA_TOOLKIT_PATH
export NCCL_INSTALL_PATH=$CUDA_TOOLKIT_PATH
export TENSORRT_INSTALL_PATH=/usr/local/tensorrt
export TF_CUDA_COMPUTE_CAPABILITIES=3.0
export TF_SET_ANDROID_WORKSPACE=0
export TF_IGNORE_MAX_BAZEL_VERSION=1
export TF_CONFIGURE_IOS=0

TENSORFLOW_VERSION=1.14.0-rc0

download https://github.com/tensorflow/tensorflow/archive/v$TENSORFLOW_VERSION.tar.gz tensorflow-$TENSORFLOW_VERSION.tar.gz

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
        if [[ -f "$P/include/python3.7m/Python.h" ]]; then
            CPYTHON_PATH="$P"
            export PYTHON_BIN_PATH="$CPYTHON_PATH/bin/python3.7"
            export PYTHON_INCLUDE_PATH="$CPYTHON_PATH/include/python3.7m/"
            export PYTHON_LIB_PATH="$CPYTHON_PATH/lib/python3.7/"
            export USE_DEFAULT_PYTHON_LIB_PATH=0
            chmod +x "$PYTHON_BIN_PATH"
        elif [[ -f "$P/include/openblas_config.h" ]]; then
            OPENBLAS_PATH="$P"
        elif [[ -f "$P/python/numpy/core/include/numpy/arrayobject.h" ]]; then
            NUMPY_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

CPYTHON_PATH="${CPYTHON_PATH//\\//}"
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"
NUMPY_PATH="${NUMPY_PATH//\\//}"

echo "Decompressing archives"
tar --totals -xzf ../tensorflow-$TENSORFLOW_VERSION.tar.gz

# Assumes Bazel is available in the path: http://bazel.io/docs/install.html
cd tensorflow-$TENSORFLOW_VERSION

# https://github.com/tensorflow/tensorflow/issues/23191
sedinplace "/distinct_host_configuration=false/d" configure.py

# Stop complaining about possibly incompatible CUDA versions
sedinplace "s/return (cudnn == cudnn_ver) and (cudart == cuda_ver)/return True/g" configure.py

# Stop the script from annoying us with Android stuff
sed -i="" "s/return has_any_rule/return True/g" configure.py

# Allow using std::unordered_map<tensorflow::string,tensorflow::checkpoint::TensorSliceSet::SliceInfo>
sed -i="" "s/const string tag/string tag/g" tensorflow/core/util/tensor_slice_set.h

# Remove comment lines containing characters that lead to encoding errors
sedinplace '/\(foo\|bar\|ops.withSubScope\)/d' tensorflow/java/src/gen/java/org/tensorflow/processor/OperatorProcessor.java

# https://github.com/tensorflow/tensorflow/issues/26155
patch -Np1 < ../../../tensorflow-cuda.patch || true

# Work around more compile issues with CUDA 10.1
sedinplace 's/constexpr auto kComputeInNHWC/auto kComputeInNHWC/g' tensorflow/core/kernels/conv_ops.cc
sedinplace 's/constexpr auto kComputeInNCHW/auto kComputeInNCHW/g' tensorflow/core/kernels/conv_ops.cc
sedinplace 's/constexpr auto get_matrix_op/auto get_matrix_op/g' tensorflow/compiler/tf2tensorrt/convert/convert_nodes.cc

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
if [[ "$EXTENSION" =~ python ]]; then
    export BUILDTARGETS="//tensorflow/tools/pip_package:build_pip_package //tensorflow/java:tensorflow"
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/"
    export PYTHONPATH="$NUMPY_PATH/python/"
    ln -sf $OPENBLAS_PATH/libopenblas.* $NUMPY_PATH/
    $PYTHON_BIN_PATH -m pip install --target=$CPYTHON_PATH/lib/python3.7/ keras_applications==1.0.6 --no-deps
    $PYTHON_BIN_PATH -m pip install --target=$CPYTHON_PATH/lib/python3.7/ keras_preprocessing==1.0.5 --no-deps
fi

case $PLATFORM in
    android-arm)
        patch -Np1 < ../../../tensorflow-android.patch
        sedinplace "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        sedinplace "s/api_level=14/api_level=21/g" WORKSPACE
        export BUILDFLAGS="--android_compiler=clang --crosstool_top=//external:android/crosstool --cpu=armeabi-v7a --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --linkopt=-s"
        ;;
    android-arm64)
        patch -Np1 < ../../../tensorflow-android.patch
        sedinplace "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        sedinplace "s/api_level=14/api_level=21/g" WORKSPACE
        export BUILDFLAGS="--android_compiler=clang --crosstool_top=//external:android/crosstool --cpu=arm64-v8a --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --linkopt=-s"
        ;;
    android-x86)
        patch -Np1 < ../../../tensorflow-android.patch
        sedinplace "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        sedinplace "s/api_level=14/api_level=21/g" WORKSPACE
        export BUILDFLAGS="--android_compiler=clang --crosstool_top=//external:android/crosstool --cpu=x86 --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --linkopt=-s"
        ;;
    android-x86_64)
        patch -Np1 < ../../../tensorflow-android.patch
        sedinplace "/    path=\"<PATH_TO_NDK>\",/c\    path=\"${ANDROID_NDK}\"," ./WORKSPACE
        sedinplace "s/api_level=14/api_level=21/g" WORKSPACE
        export BUILDFLAGS="--android_compiler=clang --crosstool_top=//external:android/crosstool --cpu=x86_64 --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --linkopt=-s"
        ;;
    linux-x86)
        patch -Np1 < ../../../tensorflow-java.patch
        # BoringSSL doesn't build on linux-x86, so disable secure grpc and leave undefined symbols
        patch -Np1 < ../../../tensorflow-unsecure.patch
        sedinplace '/-z defs/d' tensorflow/BUILD
        sedinplace "/        \":k8\": \[\":simd_x86_64\"\],/c\        \":k8\": \[\":simd_none\"\]," third_party/jpeg/BUILD.bazel
        export BUILDFLAGS="--copt=-m32 --linkopt=-m32 --linkopt=-s"
        ;;
    linux-x86_64)
        patch -Np1 < ../../../tensorflow-java.patch
        export TF_NEED_MKL=1
        export BUILDFLAGS="--config=mkl --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx `#--copt=-mavx2 --copt=-mfma` $GPU_FLAGS --action_env PYTHONPATH --copt=-m64 --linkopt=-m64 --linkopt=-s"
        export CUDA_HOME=$CUDA_TOOLKIT_PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
        if [[ -f /usr/local/cuda/bin/nvcccache ]]; then
            sedinplace "s:%{gcc_host_compiler_path}:/usr/bin/gcc:g" third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl
            sedinplace "s:%{nvcc_path}:/usr/local/cuda/bin/nvcccache:g" third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl
        fi
        ;;
    macosx-*)
        # https://github.com/tensorflow/tensorflow/issues/14174
        sedinplace 's/__align__(sizeof(T))//g' tensorflow/core/kernels/*.cu.cc
        sedinplace '/-lgomp/d' third_party/gpus/cuda/BUILD.tpl
        sedinplace 's/cp -rLf/cp -RLf/g' third_party/gpus/cuda_configure.bzl
        sedinplace 's/check_soname = version and not static/check_soname = False/g' third_party/gpus/cuda_configure.bzl
        sedinplace 's/#if __clang__/#if 0/g' tensorflow/core/util/gpu_device_functions.h
        patch -Np1 < ../../../tensorflow-java.patch
        # allows us to use ccache with Bazel
        export BAZEL_USE_CPP_ONLY_TOOLCHAIN=1
        export TF_NEED_MKL=1
        export BUILDFLAGS="--config=mkl --config=nonccl --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx `#--copt=-mavx2 --copt=-mfma` $GPU_FLAGS --action_env PYTHONPATH --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH --linkopt=-install_name --linkopt=@rpath/libtensorflow_cc.so --linkopt=-s"
        export CUDA_HOME=$CUDA_TOOLKIT_PATH
        export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
        export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}
        export PATH=$DYLD_LIBRARY_PATH:$PATH
        if [[ -f /usr/local/cuda/bin/nvcccache ]]; then
            sedinplace "s:%{gcc_host_compiler_path}:/usr/bin/gcc:g" third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl
            sedinplace "s:%{nvcc_path}:/usr/local/cuda/bin/nvcccache:g" third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl
        fi
        ;;
    windows-x86_64)
        patch -Np1 < ../../../tensorflow-java.patch
        sedinplace 's/{diff_dst_index}, diff_src_index/{(int)diff_dst_index}, (int)diff_src_index/g' tensorflow/core/kernels/mkl_relu_op.cc
        export PYTHON_BIN_PATH="C:/Program Files/Python36/python.exe"
        export BAZEL_VC="C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/"
        # try not to use /WHOLEARCHIVE as it crashes link.exe
        export NO_WHOLE_ARCHIVE_OPTION=1
        # disable __forceinline for Eigen to speed up the build
        export TF_OVERRIDE_EIGEN_STRONG_INLINE=1
        export TF_NEED_MKL=1
        export BUILDTARGETS="///tensorflow:tensorflow_static ///tensorflow/java:tensorflow"
        export BUILDFLAGS="--config=mkl --copt=//arch:AVX `#--copt=//arch:AVX2` $GPU_FLAGS --copt=//DGRPC_ARES=0 --copt=//DPB_FIELD_16BIT=1 --copt=//machine:x64 --linkopt=//machine:x64"
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
    cd bazel-tensorflow-$TENSORFLOW_VERSION/bazel-out/x64_windows-opt/
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
    cd ../../..
fi

# copy/adjust Java source files and work around loader bug in NativeLibrary.java
mkdir -p ../java
cp -r tensorflow/java/src/gen/java/* ../java
cp -r tensorflow/java/src/main/java/* ../java
cp -r tensorflow/contrib/android/java/* ../java
cp -r tensorflow/lite/java/src/main/java/* ../java
cp -r bazel-genfiles/tensorflow/java/ops/src/main/java/* ../java
sedinplace '/TensorFlow.version/d' ../java/org/tensorflow/NativeLibrary.java
# remove lines that require the Android SDK to compile
sedinplace '/Trace/d' ../java/org/tensorflow/contrib/android/TensorFlowInferenceInterface.java
# add ops files we cannot get with CMake for Windows
# patch -Np1 -d ../java < ../../../tensorflow-java-ops.patch || true

if [[ "$EXTENSION" =~ python ]]; then
    # adjust the directory structure a bit to facilitate packaging in JAR files
    ln -snf tensorflow-$TENSORFLOW_VERSION/bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow ../python
    ln -sf python/_pywrap_tensorflow_internal.so bazel-bin/tensorflow/libtensorflow_cc.so
    ln -sf external/absl_py/absl/ ../python/
    ln -sf external/astor_archive/astor/ ../python/
    ln -sf external/gast_archive/gast/ ../python/
    ln -sf external/protobuf_archive/python/google/ ../python/
    ln -sf external/six_archive/six.py ../python/
    ln -sf external/termcolor_archive/termcolor.py ../python/
    ln -sf external/wrapt/ ../python/
    $PYTHON_BIN_PATH -m pip install --target=../python/ keras_applications==1.0.6 --no-deps
    $PYTHON_BIN_PATH -m pip install --target=../python/ keras_preprocessing==1.0.5 --no-deps
fi

cd ../..
