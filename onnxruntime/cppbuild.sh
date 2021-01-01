#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" onnxruntime
    popd
    exit
fi

export CUDACXX="/usr/local/cuda/bin/nvcc"
export CUDA_HOME="/usr/local/cuda"
export CUDNN_HOME="/usr/local/cuda"
export MAKEFLAGS="-j $MAKEJ"
export PYTHON_BIN_PATH=$(which python3)
export OPENMP_FLAGS="--use_openmp"
case $PLATFORM in
    windows-*)
        if [[ -n "${CUDA_PATH:-}" ]]; then
            export CUDACXX="$CUDA_PATH/bin/nvcc"
            export CUDA_HOME="$CUDA_PATH"
            export CUDNN_HOME="$CUDA_PATH"
        fi
        export CC="cl.exe"
        export CXX="cl.exe"
        export PYTHON_BIN_PATH=$(which python.exe)
        ;;
esac

export GPU_FLAGS=
if [[ "$EXTENSION" == *gpu ]]; then
    GPU_FLAGS="--use_cuda"
fi

ONNXRUNTIME=1.6.0

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`
mkdir -p build include lib bin java

if [[ ! -d onnxruntime ]]; then
    git clone https://github.com/microsoft/onnxruntime
fi
cd onnxruntime
git reset --hard
git checkout v$ONNXRUNTIME
git submodule update --init --recursive --jobs $MAKEJ
git submodule foreach --recursive 'git reset --hard'

# work around toolchain issues on Mac and Windows
patch -p1 < ../../../onnxruntime.patch
sedinplace "s/default='Visual Studio 15 2017'/default='Ninja'/g" tools/ci_build/build.py
sedinplace 's:/Yucuda_pch.h /FIcuda_pch.h::g' cmake/onnxruntime_providers.cmake
sedinplace 's/${PROJECT_SOURCE_DIR}\/external\/cub//g' cmake/onnxruntime_providers.cmake
sedinplace 's/ CMAKE_ARGS/CMAKE_ARGS -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF/g' cmake/external/dnnl.cmake
sedinplace 's/cudnnSetRNNDescriptor(/cudnnSetRNNDescriptor_v6(/g' onnxruntime/core/providers/cuda/rnn/cudnn_rnn_base.h
sedinplace 's/HOST_NAME_MAX/sysconf(_SC_HOST_NAME_MAX)/g' onnxruntime/core/providers/cuda/cuda_call.cc
sedinplace 's/#define NO_EXCEPTION noexcept/#define NO_EXCEPTION/g' include/onnxruntime/core/session/onnxruntime_c_api.h
sedinplace 's/Provider_/_Provider_/g' onnxruntime/core/providers/shared/exported_symbols.lst

# use PTX instead of compiling for all CUDA archs to reduce library size
sedinplace 's/-gencode=arch=compute_52,code=sm_52/-arch=sm_35/g' cmake/CMakeLists.txt
sedinplace '/-gencode=arch=compute_..,code=sm_../d' cmake/CMakeLists.txt

# provide a default constructor to Ort::Value to make it more usable with std::vector
sedinplace 's/Value(std::nullptr_t)/Value(std::nullptr_t = nullptr)/g' include/onnxruntime/core/session/onnxruntime_cxx_api.h

# hack manually written JNI code and its loader to work with JavaCPP, and C++ in general
for f in java/src/main/native/*.c; do cp $f ${f}pp; done
for f in java/src/main/native/ai_onnxruntime_*.cpp; do sedinplace 's/#include "ai_onnxruntime_.*.h"/extern "C" {/g' $f; echo "}" >> $f; done
sedinplace 's/(\*jniEnv)->\(.*\)(jniEnv,/jniEnv->\1(/g' java/src/main/native/*.cpp
sedinplace 's/ WIN32/ _WIN32/g' java/src/main/native/*.cpp
sedinplace 's/FreeLibrary(/FreeLibrary((HMODULE)/g' java/src/main/native/*.cpp
sedinplace 's/(javaStrings/((jstring)javaStrings/g' java/src/main/native/*.cpp
sedinplace 's/(javaInputStrings/((jstring)javaInputStrings/g' java/src/main/native/*.cpp
sedinplace 's/(javaOutputStrings/((jstring)javaOutputStrings/g' java/src/main/native/*.cpp
sedinplace 's/return output/return (jstring)output/g' java/src/main/native/ai_onnxruntime_OnnxTensor.cpp
sedinplace 's/, carrier)/, (jobjectArray)carrier)/g' java/src/main/native/ai_onnxruntime_OnnxTensor.cpp
sedinplace 's/, dataObj)/, (jarray)dataObj)/g' java/src/main/native/ai_onnxruntime_OnnxTensor.cpp
sedinplace 's/copy = malloc/copy = (char*)malloc/g' java/src/main/native/OrtJniUtil.cpp
sedinplace 's/floatArr = malloc/floatArr = (float*)malloc/g' java/src/main/native/OrtJniUtil.cpp
sedinplace 's/Throw(javaException)/Throw((jthrowable)javaException)/g' java/src/main/native/OrtJniUtil.cpp
sedinplace '/jint JNI_OnLoad/,/}/d' java/src/main/native/OrtJniUtil.cpp
sedinplace '/static synchronized void init() throws IOException {/a\
loaded = org.bytedeco.javacpp.Loader.load(org.bytedeco.onnxruntime.presets.onnxruntime.class) != null;\
ortApiHandle = initialiseAPIBase(ORT_API_VERSION_1);\
' java/src/main/java/ai/onnxruntime/OnnxRuntime.java
sedinplace 's/return metadataJava/return (jstring)metadataJava/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp

which ctest3 &> /dev/null && CTEST="ctest3" || CTEST="ctest"
"$PYTHON_BIN_PATH" tools/ci_build/build.py --build_dir ../build --config Release --cmake_path "$CMAKE" --ctest_path "$CTEST" --build_shared_lib --use_dnnl $OPENMP_FLAGS $GPU_FLAGS

# install headers and libraries in standard directories
cp -r include/* ../include
cp -r orttraining/orttraining/models/runner/training_runner.h ../include
cp -r orttraining/orttraining/models/runner/training_util.h ../include
sedinplace '/struct ProviderInfo_OpenVINO {/,/};/d' ../include/onnxruntime/core/providers/openvino/openvino_provider_factory.h
cp -r java/src/main/java/* ../java
cp -a ../build/Release/lib* ../lib || true
cp ../build/Release/onnxruntime*.dll ../bin || true
cp ../build/Release/onnxruntime*.lib ../lib || true

# fix library with the same name for OpenMP as MKL on Mac
case $PLATFORM in
    macosx-*)
        install_name_tool -change @rpath/libomp.dylib @rpath/libiomp5.dylib ../lib/libonnxruntime.*.dylib
        ;;
esac

cd ../..
