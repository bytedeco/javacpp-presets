#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" gym
    popd
    exit
fi

GYM_VERSION=0.22.0
download https://github.com/openai/gym/archive/$GYM_VERSION.tar.gz gym-$GYM_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

CPYTHON_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/"
OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"
NUMPY_PATH="$INSTALL_PATH/../../../numpy/cppbuild/$PLATFORM/"
SCIPY_PATH="$INSTALL_PATH/../../../scipy/cppbuild/$PLATFORM/"

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
        fi
    done
    IFS="$PREVIFS"
fi

CPYTHON_PATH="${CPYTHON_PATH//\\//}"
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"
NUMPY_PATH="${NUMPY_PATH//\\//}"
SCIPY_PATH="${SCIPY_PATH//\\//}"

echo "Decompressing archives..."
tar --totals -xzf ../gym-$GYM_VERSION.tar.gz
cd gym-$GYM_VERSION

# Remove Pillow since not an actual requirement
sedinplace "s/'Pillow<=7.2.0',//g" setup.py
sedinplace '/numpy/d' setup.py

if [[ -f "$CPYTHON_PATH/include/python3.10/Python.h" ]]; then
    # setup.py won't pick up the right libgfortran.so without this
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/:$NUMPY_PATH/lib/:$SCIPY_PATH/lib/"
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

$PYTHON_BIN_PATH -m pip install --target=$PYTHON_LIB_PATH setuptools==59.1.0

# setup.py install doesn't accept absolute paths on Windows
PYTHONNOUSERSITE=1 "$PYTHON_BIN_PATH" setup.py install --prefix ..

# Adjust the directory structure a bit to facilitate packaging in JAR file
mkdir -p ../python
export MODULES=(cloudpickle `#future six pyglet` gym)
for MODULE in ${MODULES[@]}; do
    mkdir -p ../python/$MODULE.egg-info
    cp -r $PYTHON_INSTALL_PATH/$MODULE-*/$MODULE* ../python/
    cp -r $PYTHON_INSTALL_PATH/$MODULE-*/EGG-INFO/* ../python/$MODULE.egg-info/
done
#cp -r $PYTHON_INSTALL_PATH/future-*/libfuturize ../python/
#cp -r $PYTHON_INSTALL_PATH/future-*/libpasteurize ../python/
#cp -r $PYTHON_INSTALL_PATH/future-*/past ../python/
# Work around issues with pyglet
#sedinplace '/XEHeadOfExtensionList.argtypes/d' ../python/pyglet/libs/x11/xlib.py
rm -Rf $(find ../ -iname __pycache__)

cd ../..
