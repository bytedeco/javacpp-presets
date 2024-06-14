#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" pytorch
    popd
    exit
fi

export BUILD_TEST=0
export CUDACXX="/usr/local/cuda/bin/nvcc"
export CUDA_HOME="/usr/local/cuda"
export CUDNN_HOME="/usr/local/cuda"
export NCCL_ROOT="/usr/local/cuda"
export NCCL_ROOT_DIR="/usr/local/cuda"
export NCCL_INCLUDE_DIR="/usr/local/cuda/include"
export NCCL_LIB_DIR="/usr/local/cuda/lib64"
export NCCL_VERSION="2"
export MAX_JOBS=$MAKEJ
export USE_CUDA=0
export USE_CUDNN=0
export USE_NUMPY=0
export USE_OPENMP=1
export USE_SYSTEM_NCCL=1
export USE_DISTRIBUTED=1
if [[ "$EXTENSION" == *gpu ]]; then
    export USE_CUDA=1
    export USE_CUDNN=1
    export USE_FAST_NVCC=0
    export CUDA_SEPARABLE_COMPILATION=OFF
    export TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;8.0;9.0"
fi

export PYTHON_BIN_PATH=$(which python3)
if [[ $PLATFORM == windows* ]]; then
    export PYTHON_BIN_PATH=$(which python.exe)
fi

PYTORCH_VERSION=2.3.1

export PYTORCH_BUILD_VERSION="$PYTORCH_VERSION"
export PYTORCH_BUILD_NUMBER=1

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`

# Distributed needs libuv on Windows (on other platforms, it's included
# in tensorpipe)
if [[ $PLATFORM == windows* ]]; then
    if [[ ! -d libuv ]]; then
        mkdir libuv
        cd libuv
        download https://dist.libuv.org/dist/v1.39.0/libuv-v1.39.0.tar.gz libuv.tgz
        tar xfz libuv.tgz
        mkdir build
        cd build
        cmake ../libuv-v1.39.0 -DBUILD_TESTING=OFF
        cmake --build . --config Release
        cmake --install . --config Release --prefix=../dist
        cd ../..
    fi
    export libuv_ROOT=`pwd`/libuv/dist
fi

if [[ ! -d pytorch ]]; then
    git clone https://github.com/pytorch/pytorch
fi
cd pytorch
git reset --hard
git checkout v$PYTORCH_VERSION
git submodule update --init --recursive
git submodule foreach --recursive 'git reset --hard'

CPYTHON_HOST_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/host/"
CPYTHON_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/"
OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"
NUMPY_PATH="$INSTALL_PATH/../../../numpy/cppbuild/$PLATFORM/"

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ $(find "$P" -name Python.h) ]]; then
            if [[ "$(basename $P)" == "$PLATFORM_HOST" ]]; then
                CPYTHON_HOST_PATH="$P"
            fi
            if [[ "$(basename $P)" == "$PLATFORM" ]]; then
                CPYTHON_PATH="$P"
            fi
        elif [[ -f "$P/include/openblas_config.h" ]]; then
            OPENBLAS_PATH="$P"
        elif [[ -f "$P/python/numpy/_core/include/numpy/numpyconfig.h" ]]; then
            NUMPY_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

CPYTHON_HOST_PATH="${CPYTHON_HOST_PATH//\\//}"
CPYTHON_PATH="${CPYTHON_PATH//\\//}"
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"
NUMPY_PATH="${NUMPY_PATH//\\//}"

CPYTHON_PATH="$CPYTHON_HOST_PATH"
if [[ -f "$CPYTHON_PATH/include/python3.12/Python.h" ]]; then
    # setup.py won't pick up the right libgfortran.so without this
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/:$NUMPY_PATH/lib/"
    export PYTHON_BIN_PATH="$CPYTHON_PATH/bin/python3.12"
    export PYTHON_INCLUDE_PATH="$CPYTHON_PATH/include/python3.12/"
    export PYTHON_LIB_PATH="$CPYTHON_PATH/lib/python3.12/"
    export PYTHON_INSTALL_PATH="$INSTALL_PATH/lib/python3.12/site-packages/"
    export SSL_CERT_FILE="$CPYTHON_PATH/lib/python3.12/site-packages/pip/_vendor/certifi/cacert.pem"
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
$PYTHON_BIN_PATH -m pip install --target=$PYTHON_LIB_PATH setuptools==67.6.1 pyyaml==6.0.1 typing_extensions==4.8.0

case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        ;;
    macosx-arm64)
        export CC="clang"
        export CXX="clang++"
        export CMAKE_OSX_ARCHITECTURES=arm64 # enable cross-compilation on a x86_64 host machine
        export USE_MKLDNN=OFF
        export USE_QNNPACK=OFF # not compatible with arm64 as of PyTorch 2.1.2
        export CMAKE_OSX_DEPLOYMENT_TARGET=11.00 # minimum needed for arm64 support
        ;;
    macosx-x86_64)
        export CC="clang"
        export CXX="clang++"
        ;;
    windows-x86_64)
        if which ccache.exe; then
            export CC="ccache.exe cl.exe"
            export CXX="ccache.exe cl.exe"
#            export CUDAHOSTCC="cl.exe"
#            export CUDAHOSTCXX="cl.exe"
        else
            export CC="cl.exe"
            export CXX="cl.exe"
        fi
        if [[ -n "${CUDA_PATH:-}" ]]; then
            export CUDACXX="$CUDA_PATH/bin/nvcc"
            export CUDA_HOME="$CUDA_PATH"
            export CUDNN_HOME="$CUDA_PATH"
        fi
        export CFLAGS="-I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

# work around issues with the build system
sedinplace '/Werror/d' CMakeLists.txt third_party/fbgemm/CMakeLists.txt third_party/fmt/CMakeLists.txt
sedinplace 's/build_python=True/build_python=False/g' setup.py
sedinplace 's/    build_deps()/    build_deps(); sys.exit()/g' setup.py
sedinplace 's/AND NOT DEFINED ENV{CUDAHOSTCXX}//g' cmake/public/cuda.cmake
sedinplace 's/CMAKE_CUDA_FLAGS "/CMAKE_CUDA_FLAGS " --use-local-env /g' CMakeLists.txt

sedinplace 's/using ExpandingArrayDouble/public: using ExpandingArrayDouble/g' ./torch/csrc/api/include/torch/nn/options/pooling.h

# allow setting the build directory and passing CUDA options
sedinplace "s/BUILD_DIR = .build./BUILD_DIR = os.environ['BUILD_DIR'] if 'BUILD_DIR' in os.environ else 'build'/g" tools/setup_helpers/env.py
sedinplace 's/var.startswith(("BUILD_", "USE_", "CMAKE_"))/var.startswith(("BUILD_", "USE_", "CMAKE_", "CUDA_"))/g' tools/setup_helpers/cmake.py

# allow resizing std::vector<at::indexing::TensorIndex>
sedinplace 's/TensorIndex(c10::nullopt_t)/TensorIndex(c10::nullopt_t none = None)/g' aten/src/ATen/TensorIndexing.h

# add missing declarations
sedinplace '/using ExampleType = ExampleType_;/a\
  using BatchType = ChunkType;\
  using DataType = ExampleType;\
' torch/csrc/api/include/torch/data/datasets/chunk.h
sedinplace '/^};/a\
TORCH_API std::ostream& operator<<(std::ostream& stream, const nn::Module& module);\
' torch/csrc/api/include/torch/nn/module.h
sedinplace 's/char(\(.*\))/\1/g' torch/csrc/jit/serialization/pickler.h

# some windows header defines a macro named "interface"
sedinplace 's/const std::string& interface)/const std::string\& interface_name)/g' torch/csrc/distributed/c10d/ProcessGroupGloo.hpp

#USE_FBGEMM=0 USE_KINETO=0 USE_GLOO=0 USE_MKLDNN=0 \
"$PYTHON_BIN_PATH" setup.py build

rm -Rf ../lib
if [[ ! -e torch/include/gloo ]]; then
    ln -sf ../../third_party/gloo/gloo torch/include
fi
ln -sf pytorch/torch/include ../include
ln -sf pytorch/torch/lib ../lib
ln -sf pytorch/torch/bin ../bin

# fix library with correct rpath on Mac
case $PLATFORM in
    macosx-*)
        cp /usr/local/lib/libomp.dylib ../lib/libiomp5.dylib
        chmod +w ../lib/libiomp5.dylib
        install_name_tool -id @rpath/libiomp5.dylib ../lib/libiomp5.dylib
        install_name_tool -change @rpath/libomp.dylib @rpath/libiomp5.dylib ../lib/libtorch_cpu.dylib
        ;;
    windows-*)
        cp ../libuv/dist/lib/Release/* ../lib
	;;
esac

cd ../..
