#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" numpy
    popd
    exit
fi

NUMPY_VERSION=1.19.2
download https://github.com/numpy/numpy/releases/download/v$NUMPY_VERSION/numpy-$NUMPY_VERSION.tar.gz numpy-$NUMPY_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

CPYTHON_HOST_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/host/"
CPYTHON_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/"
OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ $(find "$P" -name Python.h) ]]; then
            CPYTHON_PATH="$P"
        elif [[ -f "$P/include/openblas_config.h" ]]; then
            OPENBLAS_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

CPYTHON_PATH="${CPYTHON_PATH//\\//}"
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"

echo "Decompressing archives..."
tar --totals -xzf ../numpy-$NUMPY_VERSION.tar.gz
cd numpy-$NUMPY_VERSION

echo "[openblas]"                                  > site.cfg
echo "libraries = openblas"                       >> site.cfg
echo "library_dirs = $OPENBLAS_PATH/lib/"         >> site.cfg
echo "include_dirs = $OPENBLAS_PATH/include/"     >> site.cfg

if [[ -f "$CPYTHON_PATH/include/python3.7m/Python.h" ]]; then
    # setup.py won't pick up the right libgfortran.so without this
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/"
    export PYTHON_BIN_PATH="$CPYTHON_PATH/bin/python3.7"
    export PYTHON_INCLUDE_PATH="$CPYTHON_PATH/include/python3.7m/"
    export PYTHON_LIB_PATH="$CPYTHON_PATH/lib/python3.7/"
    export PYTHON_INSTALL_PATH="$INSTALL_PATH/lib/python3.7/site-packages/"
    chmod +x "$PYTHON_BIN_PATH"
elif [[ -f "$CPYTHON_PATH/include/Python.h" ]]; then
    CPYTHON_PATH=$(cygpath $CPYTHON_PATH)
    OPENBLAS_PATH=$(cygpath $OPENBLAS_PATH)
    export PATH="$OPENBLAS_PATH:$CPYTHON_PATH:$PATH"
    export PYTHON_BIN_PATH="$CPYTHON_PATH/bin/python.exe"
    export PYTHON_INCLUDE_PATH="$CPYTHON_PATH/include/"
    export PYTHON_LIB_PATH="$CPYTHON_PATH/lib/"
    export PYTHON_INSTALL_PATH="$INSTALL_PATH/lib/site-packages/"
fi
export PYTHONPATH="$PYTHON_INSTALL_PATH"
mkdir -p "$PYTHON_INSTALL_PATH"

if ! $PYTHON_BIN_PATH -m pip install --target=$PYTHON_LIB_PATH cython; then
    echo "extra_link_args = -lgfortran"           >> site.cfg
    "$CPYTHON_HOST_PATH/bin/python3.7" -m pip install --target="$CPYTHON_HOST_PATH/lib/python3.7/" crossenv cython
    "$CPYTHON_HOST_PATH/bin/python3.7" -m crossenv "$PYTHON_BIN_PATH" crossenv
    source crossenv/bin/activate
    cross-expose cython
    PYTHON_BIN_PATH="python"
fi

case $PLATFORM in
    linux-armhf)
        ATLAS=None CC="arm-linux-gnueabihf-gcc -std=c99 -march=armv6 -mfpu=vfp -mfloat-abi=hard" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        arm-linux-gnueabihf-strip $(find ../ -iname *.so)
        ;;
    linux-arm64)
        ATLAS=None CC="aarch64-linux-gnu-gcc -mabi=lp64" CFLAGS="-O2" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        aarch64-linux-gnu-strip $(find ../ -iname *.so)
        ;;
    linux-ppc64le)
        ATLAS=None CC="powerpc64le-linux-gnu-gcc -m64" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        powerpc64le-linux-gnu-strip $(find ../ -iname *.so)
        ;;
    linux-x86)
        ATLAS=None CC="gcc -m32" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        strip $(find ../ -iname *.so)
        ;;
    linux-x86_64)
        ATLAS=None CC="gcc -m64" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        strip $(find ../ -iname *.so)
        ;;
    macosx-*)
        sedinplace 's/-std=c99/-w/g' numpy/distutils/ccompiler.py
        ATLAS=None "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        # need to add RPATH so it can find MKL in cache
        for f in $(find ../ -iname *.so); do install_name_tool -add_rpath @loader_path/../../../ $f; done
        ;;
    windows-x86)
        sedinplace '/ccompiler._default_compilers = /,+2d' numpy/distutils/ccompiler.py # don't try to use GCC
        sedinplace 's/ltype = long_double_representation(pyod("_configtest"))/ltype = "IEEE_DOUBLE_LE"/g' numpy/core/setup_common.py
        # the build sometimes fails with multiple jobs
        MAKEJ=1
        # setup.py install doesn't accept absolute paths on Windows
        ATLAS=None "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ -L$CPYTHON_PATH/libs/ install --prefix ..
        ;;
    windows-x86_64)
        # the build sometimes fails with multiple jobs
        MAKEJ=1
        # setup.py install doesn't accept absolute paths on Windows
        ATLAS=None "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ -L$CPYTHON_PATH/libs/ install --prefix ..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

ln -snf $PYTHONPATH/numpy-*/ ../python
rm -Rf $(find ../ -iname __pycache__)

cd ../..
