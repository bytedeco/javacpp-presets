#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tvm
    popd
    exit
fi

export GPU_FLAGS=
if [[ "$EXTENSION" == *gpu ]]; then
    GPU_FLAGS="-DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_CUBLAS=ON"
fi

TVM_VERSION=0.8

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`

CPYTHON_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/"
OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"
NUMPY_PATH="$INSTALL_PATH/../../../numpy/cppbuild/$PLATFORM/"
SCIPY_PATH="$INSTALL_PATH/../../../scipy/cppbuild/$PLATFORM/"
LLVM_PATH="$INSTALL_PATH/../../../llvm/cppbuild/$PLATFORM/"
MKL_PATH="$INSTALL_PATH/../../../mkl/cppbuild/$PLATFORM/"
MKLDNN_PATH="$INSTALL_PATH/../../../dnnl/cppbuild/$PLATFORM/"
OPENCL_PATH="$INSTALL_PATH/../../../opencl/cppbuild/$PLATFORM/"

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
        elif [[ -f "$P/python/scipy/version.py" ]]; then
            SCIPY_PATH="$P"
        elif [[ -f "$P/include/llvm-c/Core.h" ]]; then
            LLVM_PATH="$P"
        elif [[ -f "$P/include/mkl.h" ]]; then
            MKL_PATH="$P"
        elif [[ -f "$P/include/dnnl.h" ]]; then
            MKLDNN_PATH="$P"
        elif [[ -f "$P/include/CL/cl.h" ]]; then
            OPENCL_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

CPYTHON_PATH="${CPYTHON_PATH//\\//}"
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"
NUMPY_PATH="${NUMPY_PATH//\\//}"
SCIPY_PATH="${SCIPY_PATH//\\//}"
LLVM_PATH="${LLVM_PATH//\\//}"
MKL_PATH="${MKL_PATH//\\//}"
MKLDNN_PATH="${MKLDNN_PATH//\\//}"
OPENCL_PATH="${OPENCL_PATH//\\//}"

if [[ ! -d tvm ]]; then
    git clone https://github.com/apache/tvm
fi
cd tvm
git reset --hard
git checkout v$TVM_VERSION
git submodule update --init --recursive
git submodule foreach --recursive 'git reset --hard'

export TVM_LIBRARY_PATH=`pwd`

# prevent setuptools from trying to build NumPy or SciPy
sedinplace '/numpy/d' python/gen_requirements.py
sedinplace '/scipy/d' python/gen_requirements.py

# Fix compiler errors
sedinplace 's/uint32_t _type_child_slots_can_overflow/bool _type_child_slots_can_overflow/g' include/tvm/runtime/ndarray.h
sedinplace 's/-Werror//g' src/runtime/crt/Makefile
sedinplace '/numpy/d' python/setup.py
sedinplace '/scipy/d' python/setup.py
sedinplace '/candidate_path/d' python/setup.py

# https://github.com/apache/tvm/pull/6717
# https://github.com/apache/tvm/pull/9138
# https://github.com/apache/tvm/pull/8682
#sedinplace 's/(lanes, \/\*Scalable=\*\/false)/::getFixed(lanes)/g' src/target/llvm/codegen_llvm.cc
#sedinplace 's/llvm::Intrinsic::getName(id, {})/llvm::Intrinsic::getBaseName(id).str()/g' src/target/llvm/codegen_llvm.cc
#sedinplace 's/::F_None/::OF_None/g' src/target/llvm/llvm_module.cc

# https://github.com/apache/tvm/pull/6738
# https://github.com/apache/tvm/pull/6752
#patch -Np1 < ../../../tvm.patch

# Work around issues with llvm-config
f=($LLVM_PATH/llvm-config*)
if [[ -f $f ]]; then
    mkdir -p $LLVM_PATH/bin
    cp $LLVM_PATH/llvm-config* $LLVM_PATH/bin/
    chmod +x $LLVM_PATH/bin/llvm-config*
fi
if [[ -f "$LLVM_PATH/lib/libLLVM.dylib" ]]; then
    ln -sf libLLVM.dylib $LLVM_PATH/lib/libLLVM-13.dylib
fi
if [[ -f "$LLVM_PATH/lib/LTO.lib" ]]; then
    ln -sf LTO.lib $LLVM_PATH/lib/LLVM.lib
fi

if [[ -f "$CPYTHON_PATH/include/python3.10/Python.h" ]]; then
    # setup.py won't pick up the right libgfortran.so without this
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/:$NUMPY_PATH/lib/:$SCIPY_PATH/lib/"
    export PATH="$CPYTHON_PATH/lib/python3.10/bin/:$PATH"
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
    SCIPY_PATH=$(cygpath $SCIPY_PATH)
    export PATH="$OPENBLAS_PATH:$CPYTHON_PATH:$NUMPY_PATH:$SCIPY_PATH:$PATH"
    export PYTHON_BIN_PATH="$CPYTHON_PATH/bin/python.exe"
    export PYTHON_INCLUDE_PATH="$CPYTHON_PATH/include/"
    export PYTHON_LIB_PATH="$CPYTHON_PATH/lib/"
    export PYTHON_INSTALL_PATH="$INSTALL_PATH/lib/site-packages/"
    export SSL_CERT_FILE="$CPYTHON_PATH/lib/pip/_vendor/certifi/cacert.pem"
fi
export PYTHONPATH="$PYTHON_INSTALL_PATH:$NUMPY_PATH/python/:$SCIPY_PATH/python/"
mkdir -p "$PYTHON_INSTALL_PATH"

export CFLAGS="-I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$CPYTHON_PATH/libs/"
export PYTHONNOUSERSITE=1
$PYTHON_BIN_PATH -m pip install --target=$PYTHON_LIB_PATH setuptools==59.1.0

case $PLATFORM in
    linux-x86_64)
        $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=$MKL_PATH -DUSE_LLVM=$LLVM_PATH/bin/llvm-config -DUSE_MKLDNN=$MKLDNN_PATH -DUSE_MICRO=ON $GPU_FLAGS -DUSE_OPENCL=$OPENCL_PATH -DUSE_OPENMP=intel -DOMP_LIBRARY=$MKL_PATH/lib/libiomp5.so -DCMAKE_C_FLAGS='-Wl,-rpath,$ORIGIN/,-rpath,$ORIGIN/../../' -DCMAKE_CXX_FLAGS='-Wl,-rpath,$ORIGIN/,-rpath,$ORIGIN/../../' .
        make -j $MAKEJ
        make install/strip
        cd python
        "$PYTHON_BIN_PATH" setup.py install --prefix $INSTALL_PATH
        strip $(find ../ -iname *.so)
        cd ..
        ;;
    macosx-x86_64)
        mkdir -p ../lib
        cp /usr/local/lib/libomp.dylib ../lib/libiomp5.dylib
        chmod +w ../lib/libiomp5.dylib
        install_name_tool -id @rpath/libiomp5.dylib ../lib/libiomp5.dylib
        $CMAKE -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=$MKL_PATH -DUSE_LLVM=$LLVM_PATH/bin/llvm-config -DUSE_MKLDNN=$MKLDNN_PATH -DUSE_MICRO=ON $GPU_FLAGS -DUSE_OPENCL=$OPENCL_PATH -DUSE_OPENMP=intel -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" -DCMAKE_C_FLAGS="-I/usr/local/include -L$INSTALL_PATH/lib -liomp5" -DCMAKE_CXX_FLAGS="-I/usr/local/include -L$INSTALL_PATH/lib -liomp5" .
        make -j $MAKEJ
        make install/strip
        cd python
        "$PYTHON_BIN_PATH" setup.py install --prefix $INSTALL_PATH
        cd ..
        # need to add RPATH so it can find MKL in cache
        for f in $(find ../ -iname '*.dylib'); do install_name_tool -add_rpath @loader_path/ -add_rpath @loader_path/../../ $f || true; done
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"
        $CMAKE -G "Ninja" -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=$MKL_PATH -DUSE_LLVM=$LLVM_PATH/bin/llvm-config -DUSE_MKLDNN=$MKLDNN_PATH $GPU_FLAGS -DUSE_OPENCL=$OPENCL_PATH -DUSE_OPENMP=intel -DOMP_LIBRARY= .
        ninja -j $MAKEJ
        ninja install
        cd python
        # setup.py install doesn't accept absolute paths on Windows
        "$PYTHON_BIN_PATH" setup.py install --prefix ../..
        cd ..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

# Copy missing header files in install directory
cp -a 3rdparty/dlpack/include/dlpack 3rdparty/dmlc-core/include/dmlc ../include

# Adjust the directory structure a bit to facilitate packaging in JAR file
mkdir -p ../python
export MODULES=(attr cloudpickle decorator psutil synr typed_ast tornado tvm)
for MODULE in ${MODULES[@]}; do
    mkdir -p ../python/$MODULE.egg-info
    cp -r $PYTHON_INSTALL_PATH/$MODULE*/$MODULE* ../python/ || true
    cp -r $PYTHON_INSTALL_PATH/$MODULE*/EGG-INFO/* ../python/$MODULE.egg-info/ || true
done
rm -Rf $(find ../ -iname __pycache__)

# Copy/adjust Java source files
mkdir -p ../java
cp -r jvm/core/src/main/java/* ../java
cp -r jvm/native/src/main/native/* ../include
sedinplace '/dlfcn.h/d' ../include/org_apache_tvm_native_c_api.cc
sedinplace '/org_apache_tvm_native_c_api.h/d' ../include/org_apache_tvm_native_c_api.cc
sedinplace '/if (_tvmHandle/,/^  }/d' ../include/org_apache_tvm_native_c_api.cc
sedinplace 's/reinterpret_cast<int>/static_cast<int>/g' ../include/org_apache_tvm_native_c_api.cc
sedinplace '/#include "jni_helper_func.h"/i\
extern "C" {
' ../include/org_apache_tvm_native_c_api.cc
echo "}" >> ../include/org_apache_tvm_native_c_api.cc

cd ../..
