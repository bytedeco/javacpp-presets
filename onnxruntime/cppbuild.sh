#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" onnxruntime
    popd
    exit
fi

export ARCH_FLAGS="--allow_running_as_root"
export DNNL_FLAGS="--use_dnnl"
export CMAKE_ARGS=
export COREML_FLAGS=
export OPENMP_FLAGS= # "--use_openmp"
export CUDAFLAGS="-v"
export CUDACXX="/usr/local/cuda/bin/nvcc"
export CUDA_HOME="/usr/local/cuda"
export CUDNN_HOME="/usr/local/cuda"
export MAKEFLAGS="-j $MAKEJ"
export PYTHON_BIN_PATH=$(which python3)
export ORT_BUILD_WITH_CACHE=1

export GPU_FLAGS=
if [[ "$EXTENSION" == *gpu ]]; then
    GPU_FLAGS="--use_cuda"
fi

ONNXRUNTIME=1.23.2

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
git submodule update --init --recursive
git submodule foreach --recursive 'git reset --hard'

case $PLATFORM in
    linux-arm64)
        export CC="aarch64-linux-gnu-gcc"
        export CXX="aarch64-linux-gnu-g++"
        export ARCH_FLAGS="$ARCH_FLAGS --arm64"
        export CMAKE_ARGS="-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=AARCH64"
        ;;
    macosx-arm64)
        export ARCH_FLAGS=
        export COREML_FLAGS="--use_coreml"
        ;;
    macosx-x86_64)
        export ARCH_FLAGS=
        export COREML_FLAGS="--use_coreml"
        ;;
    windows-*)
        if [[ -n "${CUDA_PATH:-}" ]]; then
            export CUDACXX="$CUDA_PATH/bin/nvcc.exe"
            export CUDA_HOME="$CUDA_PATH"
            export CUDNN_HOME="$CUDA_PATH"
        fi
        export CC="cl.exe"
        export CXX="cl.exe"
        export ARCH_FLAGS="--cmake_generator Ninja"
        export PYTHON_BIN_PATH=$(which python.exe)
        ;;
esac

patch -p1 < ../../../onnxruntime-cuda13.patch

#if [[ -n "$ARCH_FLAGS" ]]; then
#    # build host version of protoc
#    cd ../build
#    CC= CXX= "$CMAKE" -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF ../onnxruntime/cmake/external/protobuf/cmake
#    CC= CXX= "$CMAKE" --build . --parallel $MAKEJ
#    cd ../onnxruntime
#    export ARCH_FLAGS="$ARCH_FLAGS --path_to_protoc_exe $INSTALL_PATH/build/protoc"
#fi

sedinplace 's/cmake_minimum_required(VERSION 3...)/cmake_minimum_required(VERSION 3.16)/g' cmake/CMakeLists.txt
sedinplace '/COMPILE_WARNING_AS_ERROR/d' cmake/CMakeLists.txt
sedinplace '/CMP0104/d' cmake/CMakeLists.txt
sedinplace '/Werror/d' cmake/CMakeLists.txt
sedinplace '/WX/d' cmake/CMakeLists.txt

# allow cross compilation for linux-arm64
sedinplace 's/if args.arm or args.arm64:/if False:/g' tools/ci_build/build.py
sedinplace 's/is_linux() and platform.machine() == "x86_64"/False/g' tools/ci_build/build.py
sedinplace 's/onnxruntime_target_platform STREQUAL "aarch64"/FALSE/g' cmake/CMakeLists.txt
sedinplace 's/if (NOT APPLE)/if (FALSE)/g' cmake/onnxruntime_mlas.cmake
sedinplace 's/!defined(__APPLE__)/0/g' onnxruntime/core/mlas/inc/mlas.h
sedinplace 's/defined(__aarch64__) && defined(__linux__)/0/g' `find . -name *.cpp -o -name *.cc -o -name *.h`
sedinplace 's/MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_I8MM()/false/g' onnxruntime/core/mlas/lib/platform.cpp

# work around toolchain issues on Mac and Windows
patch -p1 < ../../../onnxruntime.patch
#patch -p1 < ../../../onnxruntime-cuda.patch # https://github.com/microsoft/onnxruntime/pull/22316
#patch -p1 < ../../../onnxruntime-windows.patch # https://github.com/microsoft/onnxruntime/pull/7883
sedinplace '/--Werror/d' cmake/CMakeLists.txt
sedinplace '/-DCMAKE_CUDA_COMPILER=/d' tools/ci_build/build.py
sedinplace "s/return 'ON'/return 'OFF'/g" tools/ci_build/build.py
sedinplace "s/Visual Studio 1. 20../Ninja/g" tools/ci_build/build.py
sedinplace 's/Darwin|iOS/iOS/g' cmake/onnxruntime_providers_cpu.cmake cmake/onnxruntime_providers.cmake
sedinplace 's/-fvisibility=hidden//g' cmake/CMakeLists.txt cmake/adjust_global_compile_flags.cmake cmake/onnxruntime_providers_cpu.cmake cmake/onnxruntime_providers.cmake
sedinplace 's:/Yucuda_pch.h /FIcuda_pch.h::g' cmake/onnxruntime_providers_cuda.cmake cmake/onnxruntime_providers.cmake
sedinplace 's/${PROJECT_SOURCE_DIR}\/external\/cub//g' cmake/onnxruntime_providers_cuda.cmake cmake/onnxruntime_providers.cmake
sedinplace 's/ONNXRUNTIME_PROVIDERS_SHARED)/ONNXRUNTIME_PROVIDERS_SHARED onnxruntime_providers_shared)/g' cmake/onnxruntime_providers_cpu.cmake cmake/onnxruntime_providers.cmake
sedinplace 's/DNNL_TAG v.*)/DNNL_TAG v3.10.2)/g' cmake/external/dnnl.cmake
sedinplace 's/DNNL_SHARED_LIB libdnnl.1.dylib/DNNL_SHARED_LIB libdnnl.2.dylib/g' cmake/external/dnnl.cmake
sedinplace 's/DNNL_SHARED_LIB libdnnl.so.1/DNNL_SHARED_LIB libdnnl.so.2/g' cmake/external/dnnl.cmake
sedinplace "s/ CMAKE_ARGS/ CMAKE_ARGS $CMAKE_ARGS -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_BUILD_TESTS=OFF -DDNNL_CPU_RUNTIME=SEQ/g" cmake/external/dnnl.cmake
sedinplace 's#GIT_REPOSITORY ${DNNL_URL}#URL https://github.com/oneapi-src/oneDNN/archive/refs/tags/${DNNL_TAG}.tar.gz#g' cmake/external/dnnl.cmake
sedinplace 's/cudnnSetRNNDescriptor(/cudnnSetRNNDescriptor_v6(/g' onnxruntime/core/providers/cuda/rnn/cudnn_rnn_base.h
sedinplace 's/HOST_NAME_MAX/sysconf(_SC_HOST_NAME_MAX)/g' onnxruntime/core/providers/cuda/cuda_call.cc
sedinplace 's/#define NO_EXCEPTION noexcept/#define NO_EXCEPTION/g' include/onnxruntime/core/session/onnxruntime_c_api.h
sedinplace 's/Provider_/_Provider_/g' onnxruntime/core/providers/shared/exported_symbols.lst
sedinplace 's/ceil(/ceilf(/g' onnxruntime/core/providers/cuda/object_detection/roialign_impl.cu
sedinplace 's/ceil(/ceilf(/g' onnxruntime/core/providers/cuda/tensor/resize_impl.cu
sedinplace 's/floor(/floorf(/g' onnxruntime/core/providers/cuda/tensor/resize_impl.cu
sedinplace 's/round(/roundf(/g' onnxruntime/core/providers/cuda/tensor/resize_impl.cu
sedinplace 's/, dims_span);/);/g' onnxruntime/core/providers/dnnl/subgraph/dnnl_reduce.cc
sedinplace 's/, data_dims);/);/g' onnxruntime/core/providers/dnnl/subgraph/dnnl_squeeze.cc
sedinplace 's/, dims);/);/g' onnxruntime/contrib_ops/cuda/quantization/qordered_ops/qordered_qdq.cc
sedinplace '/omp.h/d' onnxruntime/core/providers/dnnl/dnnl_execution_provider.cc
sedinplace '/omp_get_max_threads/d' onnxruntime/core/providers/dnnl/dnnl_execution_provider.cc
sedinplace '/omp_set_num_threads/d' onnxruntime/core/providers/dnnl/dnnl_execution_provider.cc
sedinplace '/cvtfp16Avx/d' cmake/onnxruntime_mlas.cmake
sedinplace 's/MlasCastF16ToF32KernelAvx;/MlasCastF16ToF32KernelAvx2;/g' onnxruntime/core/mlas/lib/platform.cpp

# compile for all CUDA archs instead of using PTX to reduce load time
sedinplace 's/"60;70;75;80;86;89;90;100;120"/"75;80;90;100;120"/g' cmake/external/cuda_configuration.cmake
sedinplace 's/"all"/"50-real;60-real;70-real;80-real;90-real;100-real;120-real"/g' cmake/CMakeLists.txt
sedinplace 's/-gencode=arch=compute_52,code=sm_52/-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90/g' cmake/CMakeLists.txt
sedinplace '/-gencode=arch=compute_..,code=sm_../d' cmake/CMakeLists.txt

# provide a default constructor to Ort::Value to make it more usable with std::vector
sedinplace 's/Value(std::nullptr_t)/Value(std::nullptr_t = nullptr)/g' include/onnxruntime/core/session/onnxruntime_cxx_api.h

# hack manually written JNI code and its loader to work with JavaCPP, and C++ in general
for f in java/src/main/native/*.c; do cp $f ${f}pp; done
for f in java/src/main/native/ai_onnxruntime_*.cpp; do sedinplace 's/#include "ai_onnxruntime_.*.h"/extern "C" {/g' $f; echo "}" >> $f; done
sedinplace 's/(\*jniEnv)->\(.*\)([[:space:]]*jniEnv,[[:space:]]*/jniEnv->\1(/g' java/src/main/native/*.cpp
sedinplace 's/ WIN32/ _WIN32/g' java/src/main/native/*.cpp
sedinplace 's/ goto string_tensor_cleanup;/ return ORT_FAIL;/g' java/src/main/native/*.cpp
sedinplace 's/ goto .*;/ return NULL;/g' java/src/main/native/*.cpp
sedinplace 's/FreeLibrary(/FreeLibrary((HMODULE)/g' java/src/main/native/*.cpp
sedinplace 's/(javaString/((jstring)javaString/g' java/src/main/native/*.cpp
sedinplace 's/(javaInputStrings/((jstring)javaInputStrings/g' java/src/main/native/*.cpp
sedinplace 's/(javaOutputStrings/((jstring)javaOutputStrings/g' java/src/main/native/*.cpp
sedinplace 's/return output/return (jstring)output/g' java/src/main/native/ai_onnxruntime_OnnxTensor.cpp
sedinplace 's/, carrier)/, (jobjectArray)carrier)/g' java/src/main/native/ai_onnxruntime_OnnxTensor.cpp
sedinplace 's/, dataObj)/, (jarray)dataObj)/g' java/src/main/native/ai_onnxruntime_OnnxTensor.cpp
sedinplace 's/(dataObj, /((jarray)dataObj, /g' java/src/main/native/ai_onnxruntime_OnnxTensor.cpp
sedinplace 's/, dataObj, /, (jarray)dataObj, /g' java/src/main/native/ai_onnxruntime_OnnxTensor.cpp
sedinplace 's/characterBuffer = malloc/characterBuffer = (char*)malloc/g' java/src/main/native/OrtJniUtil.cpp
sedinplace 's/copy = malloc/copy = (char*)malloc/g' java/src/main/native/OrtJniUtil.cpp
sedinplace 's/floatArr = malloc/floatArr = (float*)malloc/g' java/src/main/native/OrtJniUtil.cpp
sedinplace 's/offsets = malloc/offsets = (size_t*)malloc/g' java/src/main/native/OrtJniUtil.cpp
sedinplace 's/tempBuffer = malloc/tempBuffer = (char*)malloc/g' java/src/main/native/OrtJniUtil.cpp
sedinplace 's/offsets = allocarray/offsets = (size_t*)allocarray/g' java/src/main/native/OrtJniUtil.cpp
sedinplace 's/tempBuffer = realloc/tempBuffer = (char*)realloc/g' java/src/main/native/OrtJniUtil.cpp
sedinplace 's/Throw(javaException)/Throw((jthrowable)javaException)/g' java/src/main/native/OrtJniUtil.cpp
sedinplace '/jint JNI_OnLoad/,/}/d' java/src/main/native/OrtJniUtil.cpp
sedinplace '/static synchronized void init() throws IOException {/a\
loaded = org.bytedeco.javacpp.Loader.load(org.bytedeco.onnxruntime.presets.onnxruntime.class) != null;\
ortApiHandle = initialiseAPIBase(ORT_API_VERSION_1);\
' java/src/main/java/ai/onnxruntime/OnnxRuntime.java
sedinplace 's/Names = malloc/Names = (const char**)malloc/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp
sedinplace 's/Strings = malloc/Strings = (jobject*)malloc/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp
sedinplace 's/inputValuePtrs = malloc/inputValuePtrs = (const OrtValue**)malloc/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp
sedinplace 's/outputValues = malloc/outputValues = (OrtValue**)malloc/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp
sedinplace 's/Names = allocarray/Names = (const char**)allocarray/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp
sedinplace 's/Strings = allocarray/Strings = (jobject*)allocarray/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp
sedinplace 's/inputValuePtrs = allocarray/inputValuePtrs = (const OrtValue**)allocarray/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp
sedinplace 's/outputValues = allocarray/outputValues = (OrtValue**)allocarray/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp
sedinplace 's/(\*jniEnv)->GetMethodID(/jniEnv->GetMethodID(/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp
sedinplace 's/jniEnv, metadataClazz/metadataClazz/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp
sedinplace 's/return metadataJava/return (jstring)metadataJava/g' java/src/main/native/ai_onnxruntime_OrtSession.cpp
sedinplace 's/return NULL/return/g' java/src/main/native/ai_onnxruntime_OrtSession_SessionOptions.cpp
sedinplace 's/names = allocarray/names = (const char**)allocarray/g' java/src/main/native/ai_onnxruntime_OrtSession_SessionOptions.cpp
sedinplace 's/Strings = allocarray/Strings = (jobject*)allocarray/g' java/src/main/native/ai_onnxruntime_OrtSession_SessionOptions.cpp
sedinplace 's/UTFChars(javaNameStrings/UTFChars((jstring)javaNameStrings/g' java/src/main/native/ai_onnxruntime_OrtSession_SessionOptions.cpp
sedinplace 's/initializers = allocarray/initializers = (const OrtValue**)allocarray/g' java/src/main/native/ai_onnxruntime_OrtSession_SessionOptions.cpp

which ctest3 &> /dev/null && CTEST="ctest3" || CTEST="ctest"
for i in {1..2}; do
  "$PYTHON_BIN_PATH" tools/ci_build/build.py --build_dir ../build --config Release --parallel $MAKEJ --enable_training_apis --enable_training_ops --cmake_path "$CMAKE" --ctest_path "$CTEST" --build_shared_lib $ARCH_FLAGS $DNNL_FLAGS $COREML_FLAGS $OPENMP_FLAGS $GPU_FLAGS || sedinplace 's/5ea4d05e62d7f954a46b3213f9b2535bdd866803/51982be81bbe52572b54180454df11a3ece9a934/g' cmake/deps.txt
done

# install headers and libraries in standard directories
cp -r include/* ../include
cp -r orttraining/orttraining/models/runner/training_runner.h ../include
cp -r orttraining/orttraining/models/runner/training_util.h ../include
cp -r orttraining/orttraining/training_api/include/* ../include
#sedinplace '/#include "core\/framework\/provider_options.h"/,/};/d' ../include/onnxruntime/core/providers/cuda/cuda_provider_factory.h
sedinplace '/struct ProviderInfo_OpenVINO {/,/};/d' ../include/onnxruntime/core/providers/openvino/openvino_provider_factory.h
cp -r java/src/main/jvm/* java/src/main/java/* ../java
cp -a ../build/Release/lib* ../build/Release/Release/lib* ../lib || true
cp ../build/Release/onnxruntime*.dll ../build/Release/Release/onnxruntime*.dll ../bin || true
cp ../build/Release/onnxruntime*.lib ../build/Release/Release/onnxruntime*.lib ../lib || true

# fix library with the same name for OpenMP as MKL on Mac
#case $PLATFORM in
#    macosx-*)
#        cp ../lib/libonnxruntime_providers_dnnl.so ../lib/libonnxruntime_providers_dnnl.dylib || true
#        install_name_tool -change @rpath/libomp.dylib @rpath/libiomp5.dylib ../lib/libonnxruntime.*.dylib
#        ;;
#esac

cd ../..
