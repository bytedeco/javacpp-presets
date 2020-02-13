#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" onnxruntime
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    #No Windows support yet
    echo "Error: Platform \"$PLATFORM\" is not supported"
    exit 1
fi

ONNXRUNTIME=1.1.1

mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`
mkdir -p include lib bin

if [[ ! -d onnxruntime ]]; then
    git clone https://github.com/microsoft/onnxruntime
fi
cd onnxruntime
git reset --hard
git checkout v$ONNXRUNTIME
git submodule update --init --recursive --jobs $MAKEJ
git submodule foreach --recursive git reset --hard
patch -p1 < ../../../onnxruntime.patch
which ctest3 &> /dev/null && CTEST="ctest3" || CTEST="ctest"
MAKEFLAGS="-j $MAKEJ" bash build.sh --cmake_path "$CMAKE" --ctest_path "$CTEST" --config Release --use_dnnl --build_shared_lib
sedinplace '/std::nullptr_t/d' include/onnxruntime/core/session/onnxruntime_cxx_api.h


sedinplace 's/onnxruntime_c_api.h/onnxruntime\/core\/session\/onnxruntime_c_api.h/g' include/onnxruntime/core/providers/dnnl/dnnl_provider_factory.h

sedinplace 's/: Base<OrtEnv>{p} {}/{p_ = p;}/g' include/onnxruntime/core/session/onnxruntime_cxx_api.h
sedinplace 's/: Base<OrtSessionOptions>{p} {}/{p_ = p;}/g' include/onnxruntime/core/session/onnxruntime_cxx_api.h
sedinplace 's/: Base<OrtTensorTypeAndShapeInfo>{p} {}/{p_ = p;}/g' include/onnxruntime/core/session/onnxruntime_cxx_api.h
sedinplace 's/: Base<OrtValue>{p} {}/ {p_= p;}/g' include/onnxruntime/core/session/onnxruntime_cxx_api.h
sedinplace 's/: Base<OrtTypeInfo>{p} {}/ {p_= p;}/g' include/onnxruntime/core/session/onnxruntime_cxx_api.h
sedinplace 's/: Base<OrtMemoryInfo>{p} {}/ {p_= p;}/g' include/onnxruntime/core/session/onnxruntime_cxx_api.h

#TODO: Look into restoring this, would prevent one instance of dropping to C API
sedinplace 's/Unowned<TensorTypeAndShapeInfo> GetTensorTypeAndShapeInfo() const;//g' include/onnxruntime/core/session/onnxruntime_cxx_api.h

sedinplace '/stub_api/d' include/onnxruntime/core/session/onnxruntime_cxx_api.h

sedinplace '/OrtGetApiBase/d' include/onnxruntime/core/session/onnxruntime_cxx_api.h

sedinplace '/delete/d' include/onnxruntime/core/session/onnxruntime_cxx_api.h

sedinplace '/s_api/d' include/onnxruntime/core/session/onnxruntime_cxx_api.h

sedinplace 's/std::string&&/std::string/g' include/onnxruntime/core/session/onnxruntime_cxx_api.h

sedinplace '/inline Unowned<TensorTypeAndShapeInfo> TypeInfo/,+4d' include/onnxruntime/core/session/onnxruntime_cxx_inline.h

cp -r include/* ../include
cp -r build/Linux/Release/lib* build/Linux/Release/dnnl/install/lib*/libdnnl* ../lib

cd ../..
