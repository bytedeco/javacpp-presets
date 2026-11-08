#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" pytorch
    popd
    exit
fi

export BUILD_TEST=0
export BUILD_PYTHON=OFF
#export CUDAHOSTCC="clang"
#export CUDAHOSTCXX="clang++"
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
export USE_NCCL=0
export USE_XPU=0
export USE_NUMPY=0
export USE_OPENMP=1
export USE_SYSTEM_NCCL=1
export USE_DISTRIBUTED=1

if [[ "$EXTENSION" == *gpu ]]; then
    export USE_CUDA=1
    export USE_CUDNN=1
    export USE_NCCL=1
    export USE_FAST_NVCC=0
    export CUDA_SEPARABLE_COMPILATION=OFF
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0;10.0;12.0"
fi

export PYTHON_BIN_PATH=$(which python3)
if [[ $PLATFORM == windows* ]]; then
    export PYTHON_BIN_PATH=$(which python.exe)
fi

PYTORCH_VERSION=2.15.0

export PYTORCH_BUILD_VERSION="$PYTORCH_VERSION"
export PYTORCH_BUILD_NUMBER=1

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`

# Distributed needs libuv on Windows (on other platforms, it's included in tensorpipe)
if [[ $PLATFORM == windows* ]]; then
    if [[ ! -d libuv ]]; then
        mkdir libuv
        cd libuv
        download https://dist.libuv.org/dist/v1.39.0/libuv-v1.39.0.tar.gz libuv.tgz
        tar xfz libuv.tgz
        mkdir build
        cd build
        export CC="cl.exe"
        export CXX="cl.exe"
        # libuv 1.39 needs its policy version declared for recent CMake
        cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ../libuv-v1.39.0 -DBUILD_TESTING=OFF
        cmake --build . --config Release
        cmake --install . --config Release --prefix ../dist
        cd ../..
    fi
    export libuv_ROOT=${INSTALL_PATH}/libuv/dist
fi

if [[ ! -d pytorch ]]; then
    git clone https://github.com/pytorch/pytorch
fi

cd pytorch
git reset --hard
git checkout v$PYTORCH_VERSION
git submodule update --init --recursive
git submodule foreach --recursive 'git reset --hard'

# https://github.com/pytorch/pytorch/pull/158184
# https://github.com/pytorch/pytorch/pull/159869
#patch -Np1 < ../../../pytorch.patch

# https://github.com/pytorch/pytorch/pull/164570
#patch -Np1 < ../../../pytorch-cuda.patch

# https://github.com/pytorch/pytorch/pull/189704
#patch -Np1 < ../../../pytorch-macosx.patch

CPYTHON_HOST_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/host/"
CPYTHON_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/"
OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"
NUMPY_PATH="$INSTALL_PATH/../../../numpy/cppbuild/$PLATFORM/"
# a native build has no host CPython, so let it be resolved from BUILD_PATH or below
if [[ "${PLATFORM_HOST:-$PLATFORM}" == "$PLATFORM" ]]; then
    unset CPYTHON_HOST_PATH
fi

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

export OpenBLAS_HOME=$OPENBLAS_PATH

CPYTHON_HOST_PATH="${CPYTHON_HOST_PATH:-}"
CPYTHON_HOST_PATH="${CPYTHON_HOST_PATH//\\//}"
CPYTHON_PATH="${CPYTHON_PATH//\\//}"
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"
NUMPY_PATH="${NUMPY_PATH//\\//}"

# the target CPython plays the host role when no host build was provided
if [[ -z "${CPYTHON_HOST_PATH:-}" ]]; then
    CPYTHON_HOST_PATH="$CPYTHON_PATH"
fi

CPYTHON_PATH="$CPYTHON_HOST_PATH"
if [[ -f "$CPYTHON_PATH/include/python3.15/Python.h" ]]; then
    # setup.py won't pick up the right libgfortran.so without this
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/:$NUMPY_PATH/lib/"
    export PYTHON_BIN_PATH="$CPYTHON_PATH/bin/python3.15"
    export PYTHON_INCLUDE_PATH="$CPYTHON_PATH/include/python3.15/"
    export PYTHON_LIB_PATH="$CPYTHON_PATH/lib/python3.15/"
    export PYTHON_INSTALL_PATH="$INSTALL_PATH/lib/python3.15/site-packages/"
    export SSL_CERT_FILE="$CPYTHON_PATH/lib/python3.15/site-packages/pip/_vendor/certifi/cacert.pem"
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

export CFLAGS="-I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/"
export PYTHONNOUSERSITE=1
if [[ $PLATFORM == windows-arm64 ]]; then
    printf '%s\n' 'Cython==3.2.6' > cython-build-constraints.txt
    PIP_NO_BINARY=cython PIP_NO_CACHE_DIR=1 NO_CYTHON_COMPILE=true PIP_CONSTRAINT="$(cygpath -m "$PWD/cython-build-constraints.txt")" "$PYTHON_BIN_PATH" -m pip install --target="$PYTHON_LIB_PATH" setuptools==67.6.1 pyyaml==6.0.2 packaging==25.0 scikit-build-core==1.0.3 six==1.17.0
else
    $PYTHON_BIN_PATH -m pip install --target=$PYTHON_LIB_PATH setuptools==67.6.1 pyyaml==6.0.2 packaging==25.0 scikit-build-core==1.0.3 six==1.17.0
fi

"$PYTHON_BIN_PATH" -m pip install --upgrade --target="$PYTHON_LIB_PATH" typing_extensions==4.16.0

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
        # export PATH=$(brew --prefix llvm@18)/bin:$PATH # Use brew LLVM instead of Xcode LLVM 14
        export USE_MKLDNN=OFF
        export USE_QNNPACK=OFF # not compatible with arm64 as of PyTorch 2.1.2
        export CMAKE_OSX_DEPLOYMENT_TARGET=11.00 # minimum needed for arm64 support
        ;;
    macosx-x86_64)
        export CC="clang"
        export CXX="clang++"
        export LDFLAGS="-undefined dynamic_lookup"
        export USE_MKLDNN=OFF
        sedinplace '/CHECK_FUNCTION_EXISTS(malloc_usable_size HAVE_MALLOC_USABLE_SIZE)/d' cmake/Dependencies.cmake aten/src/ATen/CMakeLists.txt
        sedinplace '/CHECK_FUNCTION_EXISTS(posix_fallocate HAVE_POSIX_FALLOCATE)/d' cmake/Dependencies.cmake aten/src/ATen/CMakeLists.txt
        # export PATH=$(brew --prefix llvm@18)/bin:$PATH # Use brew LLVM instead of Xcode LLVM 14
        ;;
    windows-arm64 | windows-x86_64)
        # the MSVC and Windows SDK directories go in front of /usr/bin, whose coreutils
        # link.EXE and a stray ccache.exe CMake otherwise picks over cl.exe and the real
        # linker, while the rest of PATH stays in the native form that vcvarsall set up
        CL_DIR=$(dirname "$(which cl.exe)")
        WIN_KIT=$(find "$(cygpath -u "${WindowsSDKDir:-${WINDOWSSDKDIR:-}}/bin")" -maxdepth 1 -name "10.*" -type d 2>/dev/null | sort | tail -1)
        NEW_PATH=""
        PREVP="$PATH"
        PREVA=$IFS
        IFS=":"
        for P in $PREVP; do
            NP=$P
            case $(basename "$P") in
                usr\bin|bin|usr/sbin|local\\bin) ;; # keep MSYS/bash dirs in POSIX form
                *) NP=$(cygpath -u "$P" 2>/dev/null || echo "$P");;
            esac
            NEW_PATH="${NEW_PATH:+$NEW_PATH:}$NP"
        done
        IFS="$PREVA"
        # keep only the MSVC directories in native form for CMake's environment detection
        CL_WIN=$(cygpath -w "$CL_DIR")
        SDK_WIN=""
        WINDOWS_KIT_ARCHITECTURE=x64
        [[ $PLATFORM == windows-arm64 ]] && WINDOWS_KIT_ARCHITECTURE=arm64
        [[ -n "$WIN_KIT" ]] && SDK_WIN=$(cygpath -w "$WIN_KIT/$WINDOWS_KIT_ARCHITECTURE")
        export PATH="$CL_WIN${SDK_WIN:+:$SDK_WIN}:$NEW_PATH"
        # drop any directory that contains only a stray ccache.exe (no cl.exe next to it)
        CCACHE_PATH=$(command -v ccache.exe 2>/dev/null || true)
        if [[ -n "$CCACHE_PATH" && ! -f "$(dirname "$CCACHE_PATH")/cl.exe" ]]; then
            CDIR_WIN=$(cygpath -w "$(dirname "$CCACHE_PATH")")
            CDIR_MSY=$(cygpath -u "$(dirname "$CCACHE_PATH")")
            export PATH="${PATH//:$CDIR_WIN:/:}"
            export PATH="${PATH//:$CDIR_MSY:/:}"
            case $PATH in "$CDIR_WIN:"*) PATH="${PATH#"$CDIR_WIN:"};; esac
            case $PATH in *":$CDIR_WIN") PATH="${PATH%":$CDIR_WIN"}";; esac
            case $PATH in "$CDIR_MSY:"*) PATH="${PATH#"$CDIR_MSY:"};; esac
            case $PATH in *":$CDIR_MSY") PATH="${PATH%":$CDIR_MSY"}";; esac
            echo "Stripped stray ccache directory from PATH: $CCACHE_PATH"
        fi
        # ccache must sit next to cl.exe to be usable, as with the MSYS2 one
        CCACHE_OK=0
        if command -v ccache.exe >/dev/null 2>&1 && [[ "$(dirname "$(command -v ccache.exe)")" == "$(dirname "$(command -v cl.exe)")" ]]; then
            CCACHE_OK=1
        fi
        if [[ $CCACHE_OK == 1 ]]; then
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
        export USE_NCCL=0 # Not supported on Windows
        if [[ $PLATFORM == windows-arm64 ]]; then
            export USE_MKLDNN=OFF # no Windows on ARM support in oneDNN
            export USE_FBGEMM=OFF # no Windows on ARM support in FBGEMM
            export USE_KINETO=OFF # no Windows on ARM support in Kineto
            export USE_XNNPACK=OFF # no Windows on ARM support in XNNPACK
            export USE_NUMPY=1 # numpy headers/libs built above against this CPython
            # SLEEF's SVE probe passes with MSVC, but helpersve.h requires __ARM_FEATURE_SVE.
            sedinplace '/add_subdirectory(.*third_party\/sleef/i\
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")\
      set(SLEEF_DISABLE_SVE ON CACHE BOOL "Disable SVE for Windows ARM64 MSVC" FORCE)\
    endif()\
' aten/src/ATen/CMakeLists.txt
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

#sedinplace 's/,code=sm_.*)/,code=compute_60)/g' cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake

# work around issues with the build system
sedinplace '/set(CMAKE_CXX_STANDARD/i\
    set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
' CMakeLists.txt
sedinplace '/Werror/d' CMakeLists.txt third_party/fbgemm/CMakeLists.txt third_party/fmt/CMakeLists.txt
sedinplace '/setuptools.command.bdist_wheel/d' setup.py
sedinplace 's/build_python=True/build_python=False/g' setup.py
sedinplace 's/    build_deps()/    build_deps(); sys.exit()/g' setup.py
sedinplace 's/AND NOT DEFINED ENV{CUDAHOSTCXX}//g' cmake/public/cuda.cmake
sedinplace 's/CMAKE_CUDA_FLAGS "/CMAKE_CUDA_FLAGS " --use-local-env /g' CMakeLists.txt
sedinplace 's/-Xcompiler  \/Zc:__cplusplus/-Xcompiler  \/Zc:__cplusplus -Xcompiler \/Zc:preprocessor/g' CMakeLists.txt
sedinplace 's/FMT_USE_CMAKE_MODULES TRUE/FMT_USE_CMAKE_MODULES FALSE/g' third_party/fmt/CMakeLists.txt #third_party/kineto/libkineto/third_party/fmt/CMakeLists.txt

#sedinplace '/pycore_opcode.h/d' torch/csrc/dynamo/cpython_defs.c functorch/csrc/dim/dim*
sedinplace 's/using ExpandingArrayDouble/public: using ExpandingArrayDouble/g' ./torch/csrc/api/include/torch/nn/options/pooling.h

# allow setting the build directory and passing CUDA options
#sedinplace "s/BUILD_DIR = .build./BUILD_DIR = os.environ['BUILD_DIR'] if 'BUILD_DIR' in os.environ else 'build'/g" tools/setup_helpers/env.py
#sedinplace 's/var.startswith(("BUILD_", "USE_", "CMAKE_"))/var.startswith(("BUILD_", "USE_", "CMAKE_", "CUDA_"))/g' tools/setup_helpers/cmake.py

# allow resizing std::vector<at::indexing::TensorIndex> and std::vector<torch::optim::OptimizerParamGroup>
sedinplace 's/TensorIndex(c10::nullopt_t.*)/TensorIndex(c10::nullopt_t none = None)/g' aten/src/ATen/TensorIndexing.h
sedinplace 's/TensorIndex(std::nullopt_t.*)/TensorIndex(std::nullopt_t none = None)/g' aten/src/ATen/TensorIndexing.h
sedinplace '/OptimizerParamGroup& operator=(const OptimizerParamGroup& param_group) =/{N;d;}' torch/csrc/api/include/torch/optim/optimizer.h
sedinplace '/OptimizerParamGroup& operator=(OptimizerParamGroup&& param_group)/i\
  OptimizerParamGroup() {}\
  OptimizerParamGroup& operator=(const OptimizerParamGroup& param_group) {\
      params_ = param_group.params();\
      if (param_group.has_options())\
          options_ = param_group.options().clone();\
      return *this;\
  }\
' torch/csrc/api/include/torch/optim/optimizer.h

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

# fix missing #include (Pytorch 2.4.0)
sedinplace 's/#include <stdexcept>/#include <stdexcept>\
#include <vector>\
#include <unordered_map>/'  torch/csrc/distributed/c10d/control_plane/Handlers.cpp

# Remove pytorch adaptations of FindOpenMP.cmake that.
# On Windows without iomp and with new versions of VS 2019, including -openmp:experimental and libomp, causes
# final binary to be linked to both libomp and vcomp and produce incorrect results.
# Wait for eventual upstream fix, or for cmake 2.30 that allows to choose between -openmp and -openmp:experimental
# and see if choosing experimental works. See Issue #1503.
# On Linux, pytorch FindOpenMP.cmake picks llvm libomp over libgomp. See Issue #1504.
# On MacOS CMake standard version works tooL
sedinplace '/^[[:space:]]*include(.*FindOpenMP[.]cmake)/c\  find_package(OpenMP)' cmake/Dependencies.cmake
if grep -q 'include(.*FindOpenMP[.]cmake)' cmake/Dependencies.cmake || ! grep -q 'find_package(OpenMP)' cmake/Dependencies.cmake; then
    echo "Error: Failed to replace the PyTorch OpenMP module include"
    exit 1
fi
rm -f cmake/Modules/FindOpenMP.cmake

# delete broken CUDA kernels at least on Windows with CUDA 12.9
#rm -f aten/src/ATen/native/cuda/SegmentReduce.cu

# clear any stale CMake cache left by a failed configure run: it pins the wrong compiler
# (e.g. a stray ccache) and CMake keeps reusing it without a fresh configure
rm -rf build/CMakeCache.txt build/CMakeFiles

#USE_FBGEMM=0 USE_KINETO=0 USE_GLOO=0 USE_MKLDNN=0 \
BLAS=OpenBLAS "$PYTHON_BIN_PATH" setup.py develop

rm -Rf ../include ../lib ../bin
#if [[ ! -e torch/include/gloo ]]; then
#    ln -sf ../../third_party/gloo/gloo torch/include
#fi
ln -sf pytorch/build/include ../include
ln -sf pytorch/build/lib ../lib
ln -sf pytorch/build/bin ../bin
rm -f ../lib/libomp.dylib

case $PLATFORM in
    macosx-arm64)
        cp "$(brew ls libomp|grep libomp.dylib)" ../lib/
        ;;
    macosx-x86_64)
        # Disguise libomp as libiomp5 (they share the same codebase and have the same symbols)
        # This helps if user wants to link with MKL.
        # On linux, user linking with mkl would need to set
        # MKL_THREADING_LAYER=GNU
        cp "$(brew ls libomp|grep libomp.dylib)" ../lib/libiomp5.dylib
        chmod +w ../lib/libiomp5.dylib
        install_name_tool -id @rpath/libiomp5.dylib ../lib/libiomp5.dylib
        codesign --force -s - ../lib/libiomp5.dylib
        old=$(otool -L ../lib/libtorch_cpu.dylib|grep libomp.dylib|awk '{print $1}')
        echo install_name_tool -change $old @rpath/libiomp5.dylib ../lib/libtorch_cpu.dylib
        install_name_tool -change $old @rpath/libiomp5.dylib ../lib/libtorch_cpu.dylib
        codesign --force -s - ../lib/libtorch_cpu.dylib
        ;;
    windows-*)
        cp ../libuv/dist/lib/Release/* ../lib
	;;
esac

cd ../..
