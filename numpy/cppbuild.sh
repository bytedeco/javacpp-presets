#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" numpy
    popd
    exit
fi

NUMPY_VERSION=1.16.2
download https://github.com/numpy/numpy/releases/download/v$NUMPY_VERSION/numpy-$NUMPY_VERSION.tar.gz numpy-$NUMPY_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

CPYTHON_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/"
OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -f "$P/include/Python.h" ]]; then
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

case $PLATFORM in
    linux-*)
        # setup.py won't pick up the right libgfortran.so.4 without this
        export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/"
        export PYTHONPATH=$INSTALL_PATH/lib64/python3.6/site-packages
        mkdir -p $PYTHONPATH
        python3 setup.py build -j $MAKEJ install --prefix $INSTALL_PATH
        strip $(find ../ -iname *.so)
        ;;
    macosx-*)
        export PYTHONPATH=$INSTALL_PATH/lib/python3.6/site-packages
        mkdir -p $PYTHONPATH
        python3 setup.py build -j $MAKEJ install --prefix $INSTALL_PATH
        ;;
    windows-*)
        export PYTHONPATH=$INSTALL_PATH/lib/site-packages
        mkdir -p $PYTHONPATH
        # setup.py doesn't accept absolute paths on Windows
        "/C/Program Files/Python36/python" setup.py build -j $MAKEJ install --prefix ..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

ln -snf $PYTHONPATH/numpy-*/ ../python
rm -Rf $(find ../ -iname __pycache__)

cd ../..
