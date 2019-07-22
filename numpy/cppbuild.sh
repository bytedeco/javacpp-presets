#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" numpy
    popd
    exit
fi

NUMPY_VERSION=1.17.0rc2
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

case $PLATFORM in
    linux-*)
        # setup.py won't pick up the right libgfortran.so.4 without this
        export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/"
        export PYTHONPATH="$INSTALL_PATH/lib/python3.7/site-packages/"
        mkdir -p "$PYTHONPATH"
        chmod +x "$CPYTHON_PATH/bin/python3.7"
        "$CPYTHON_PATH/bin/python3.7" setup.py build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        strip $(find ../ -iname *.so)
        ;;
    macosx-*)
        export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/"
        export PYTHONPATH="$INSTALL_PATH/lib/python3.7/site-packages/"
        mkdir -p "$PYTHONPATH"
        chmod +x "$CPYTHON_PATH/bin/python3.7"
        "$CPYTHON_PATH/bin/python3.7" setup.py build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        # need to add RPATH so it can find MKL in cache
        for f in $(find ../ -iname *.so); do install_name_tool -add_rpath @loader_path/../../../ $f; done
        ;;
    windows-*)
        CPYTHON_PATH=$(cygpath $CPYTHON_PATH)
        OPENBLAS_PATH=$(cygpath $OPENBLAS_PATH)
        export PATH="$PATH:$OPENBLAS_PATH/:$CPYTHON_PATH/"
        export PYTHONPATH="$INSTALL_PATH/lib/site-packages/"
        mkdir -p "$PYTHONPATH"
        # the build sometimes fails with multiple jobs
        MAKEJ=1
        # setup.py install doesn't accept absolute paths on Windows
        "$CPYTHON_PATH/bin/python.exe" setup.py build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ -L$CPYTHON_PATH/libs/ install --prefix ..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

ln -snf $PYTHONPATH/numpy-*/ ../python
rm -Rf $(find ../ -iname __pycache__)

cd ../..
