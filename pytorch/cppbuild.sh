#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" pytorch
    popd
    exit
fi

export BUILD_TEST=0
export MAX_JOBS=$MAKEJ
export USE_CUDA=0
export USE_NUMPY=0
export USE_OPENMP=1
if [[ "$EXTENSION" == *gpu ]]; then
    export USE_CUDA=1
    export USE_FAST_NVCC=0
    export CUDA_SEPARABLE_COMPILATION=OFF
    export TORCH_CUDA_ARCH_LIST="3.5+PTX"
fi

PYTORCH_VERSION=1.10.2

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`

if [[ ! -d pytorch ]]; then
    git clone https://github.com/pytorch/pytorch
fi
cd pytorch
git reset --hard
git checkout v$PYTORCH_VERSION
git submodule update --init --recursive
git submodule foreach --recursive 'git reset --hard'

# https://github.com/pytorch/pytorch/pull/66219
patch -Np1 < ../../../pytorch.patch

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

CPYTHON_PATH="${CPYTHON_PATH//\\//}"
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"
NUMPY_PATH="${NUMPY_PATH//\\//}"

if [[ -f "$CPYTHON_PATH/include/python3.10/Python.h" ]]; then
    # setup.py won't pick up the right libgfortran.so without this
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/:$NUMPY_PATH/lib/"
    export PYTHON_BIN_PATH="$CPYTHON_PATH/bin/python3.10"
    export PYTHON_INCLUDE_PATH="$CPYTHON_PATH/include/python3.10/"
    export PYTHON_LIB_PATH="$CPYTHON_PATH/lib/python3.10/"
    export PYTHON_INSTALL_PATH="$INSTALL_PATH/lib/python3.10/site-packages/"
    export SSL_CERT_FILE="$CPYTHON_PATH/lib/python3.10/site-packages/pip/_vendor/certifi/cacert.pem"
    chmod +x "$PYTHON_BIN_PATH"
elif [[ -f "$CPYTHON_PATH/include/Python.h" ]]; then
    CPYTHON_PATH=$(cygpath $CPYTHON_PATH)
    OPENBLAS_PATH=$(cygpath $OPENBLAS_PATH)
    NUMPY_PATH=$(cygpath $NUMPY_PATH)
    export PATH="$OPENBLAS_PATH:$CPYTHON_PATH:$NUMPY_PATH:$PATH"
    export PYTHON_BIN_PATH="$CPYTHON_PATH/bin/python.exe"
    export PYTHON_INCLUDE_PATH="$CPYTHON_PATH/include/"
    export PYTHON_LIB_PATH="$CPYTHON_PATH/lib/"
    export PYTHON_INSTALL_PATH="$INSTALL_PATH/lib/site-packages/"
    export SSL_CERT_FILE="$CPYTHON_PATH/lib/pip/_vendor/certifi/cacert.pem"
fi
export PYTHONPATH="$PYTHON_INSTALL_PATH:$NUMPY_PATH/python/"
mkdir -p "$PYTHON_INSTALL_PATH"

export CFLAGS="-I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$CPYTHON_PATH/libs/"
export PYTHONNOUSERSITE=1
$PYTHON_BIN_PATH -m pip install --target=$PYTHON_LIB_PATH setuptools==59.1.0 pyyaml==6.0 typing_extensions==4.0.0

case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        ;;
    macosx-*)
        ln -sf pytorch/torch/lib ../lib
        cp /usr/local/lib/libomp.dylib ../lib/libiomp5.dylib
        chmod +w ../lib/libiomp5.dylib
        install_name_tool -id @rpath/libiomp5.dylib ../lib/libiomp5.dylib
        export CC="clang -L$INSTALL_PATH/lib -Wl,-rpath,$INSTALL_PATH/lib -liomp5 -Wno-unused-command-line-argument"
        export CXX="clang++ -L$INSTALL_PATH/lib -Wl,-rpath,$INSTALL_PATH/lib -liomp5 -Wno-unused-command-line-argument"
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"
        export CFLAGS="-I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

sedinplace '/Werror/d' CMakeLists.txt
sedinplace 's/build_python=True/build_python=False/g' setup.py
sedinplace 's/    build_deps()/    build_deps(); sys.exit()/g' setup.py

# work around some compiler bugs
sedinplace 's/!defined(__INTEL_COMPILER))/!defined(__INTEL_COMPILER) \&\& (__GNUC__ < 11))/g' third_party/XNNPACK/src/xnnpack/intrinsics-polyfill.h
sedinplace 's/using ExpandingArrayDouble/public: using ExpandingArrayDouble/g' ./torch/csrc/api/include/torch/nn/options/pooling.h
sedinplace 's/typedef c10::variant/public: typedef c10::variant/g' ./torch/csrc/api/include/torch/nn/options/upsampling.h
sedinplace 's/std::copysign/copysignf/g' aten/src/ATen/native/cuda/*.cu
sedinplace 's/std::trunc/truncf/g' aten/src/ATen/native/cuda/*.cu
sedinplace 's/std::floor/floorf/g' aten/src/ATen/native/cuda/*.cu
sedinplace 's/std::ceil/ceilf/g' aten/src/ATen/native/cuda/*.cu
sedinplace 's/round(/roundf(/g' aten/src/ATen/native/cuda/*.cu
sedinplace 's/floor(/floorf(/g' aten/src/ATen/native/cuda/*.cu
sedinplace 's/ceil(/ceilf(/g' aten/src/ATen/native/cuda/*.cu
sedinplace '/#include <thrust\/device_vector.h>/a\
#include <thrust\/host_vector.h>\
' caffe2/utils/math_gpu.cu

# allow setting the build directory and passing CUDA options
sedinplace "s/BUILD_DIR = 'build'/BUILD_DIR = os.environ['BUILD_DIR'] if 'BUILD_DIR' in os.environ else 'build'/g" tools/setup_helpers/env.py
sedinplace "s/var.startswith(('BUILD_', 'USE_', 'CMAKE_'))/var.startswith(('BUILD_', 'USE_', 'CMAKE_', 'CUDA_'))/g" tools/setup_helpers/cmake.py

# allow resizing std::vector<at::indexing::TensorIndex>
sedinplace 's/TensorIndex(c10::nullopt_t)/TensorIndex(c10::nullopt_t none = None)/g' aten/src/ATen/TensorIndexing.h

# add missing declarations
sedinplace '/^};/a\
TORCH_API std::ostream& operator<<(std::ostream& stream, const nn::Module& module);\
' torch/csrc/api/include/torch/nn/module.h

"$PYTHON_BIN_PATH" setup.py build

rm -Rf ../lib
ln -sf pytorch/torch/include ../include
ln -sf pytorch/torch/lib ../lib
ln -sf pytorch/torch/bin ../bin

cd ../..
