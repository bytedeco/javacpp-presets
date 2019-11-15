#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" numpy
    popd
    exit
fi

NUMPY_VERSION=1.17.4
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

case $PLATFORM in
    linux-armhf)
        # setup.py won't pick up the right libgfortran.so without this
        export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/"
        export PYTHONPATH="$INSTALL_PATH/lib/python3.7/site-packages/"
        mkdir -p "$PYTHONPATH"
        "$CPYTHON_HOST_PATH/bin/python3.7" -m pip install --target="$CPYTHON_HOST_PATH/lib/python3.7/" crossenv
        "$CPYTHON_HOST_PATH/bin/python3.7" -m crossenv "$CPYTHON_PATH/bin/python3.7" crossenv
        source crossenv/bin/activate
        ATLAS=None CC="arm-linux-gnueabihf-gcc -std=c99 -march=armv6 -mfpu=vfp -mfloat-abi=hard" python setup.py build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        arm-linux-gnueabihf-strip $(find ../ -iname *.so)
        ;;
    linux-arm64)
        # setup.py won't pick up the right libgfortran.so without this
        export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/"
        export PYTHONPATH="$INSTALL_PATH/lib/python3.7/site-packages/"
        mkdir -p "$PYTHONPATH"
        "$CPYTHON_HOST_PATH/bin/python3.7" -m pip install --target="$CPYTHON_HOST_PATH/lib/python3.7/" crossenv
        "$CPYTHON_HOST_PATH/bin/python3.7" -m crossenv "$CPYTHON_PATH/bin/python3.7" crossenv
        source crossenv/bin/activate
        ATLAS=None CC="aarch64-linux-gnu-gcc -mabi=lp64" python setup.py build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        aarch64-linux-gnu-strip $(find ../ -iname *.so)
        ;;
    linux-ppc64le)
        # setup.py won't pick up the right libgfortran.so without this
        export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/"
        export PYTHONPATH="$INSTALL_PATH/lib/python3.7/site-packages/"
        mkdir -p "$PYTHONPATH"
        "$CPYTHON_HOST_PATH/bin/python3.7" -m pip install --target="$CPYTHON_HOST_PATH/lib/python3.7/" crossenv
        "$CPYTHON_HOST_PATH/bin/python3.7" -m crossenv "$CPYTHON_PATH/bin/python3.7" crossenv
        source crossenv/bin/activate
        ATLAS=None CC="powerpc64le-linux-gnu-gcc -m64" python setup.py build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        powerpc64le-linux-gnu-strip $(find ../ -iname *.so)
        ;;
    linux-x86)
        # setup.py won't pick up the right libgfortran.so without this
        export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/"
        export PYTHONPATH="$INSTALL_PATH/lib/python3.7/site-packages/"
        mkdir -p "$PYTHONPATH"
        chmod +x "$CPYTHON_PATH/bin/python3.7"
        ATLAS=None CC="gcc -m32" "$CPYTHON_PATH/bin/python3.7" setup.py build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        strip $(find ../ -iname *.so)
        ;;
    linux-x86_64)
        # setup.py won't pick up the right libgfortran.so without this
        export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/"
        export PYTHONPATH="$INSTALL_PATH/lib/python3.7/site-packages/"
        mkdir -p "$PYTHONPATH"
        chmod +x "$CPYTHON_PATH/bin/python3.7"
        ATLAS=None CC="gcc -m64" "$CPYTHON_PATH/bin/python3.7" setup.py build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        strip $(find ../ -iname *.so)
        ;;
    macosx-*)
        export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/"
        export PYTHONPATH="$INSTALL_PATH/lib/python3.7/site-packages/"
        mkdir -p "$PYTHONPATH"
        chmod +x "$CPYTHON_PATH/bin/python3.7"
        ATLAS=None "$CPYTHON_PATH/bin/python3.7" setup.py build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ install --prefix $INSTALL_PATH
        # need to add RPATH so it can find MKL in cache
        for f in $(find ../ -iname *.so); do install_name_tool -add_rpath @loader_path/../../../ $f; done
        ;;
    windows-x86)
        sedinplace '/ccompiler._default_compilers = /,+2d' numpy/distutils/ccompiler.py # don't try to use GCC
        sedinplace 's/ltype = long_double_representation(pyod("_configtest"))/ltype = "IEEE_DOUBLE_LE"/g' numpy/core/setup_common.py
        CPYTHON_PATH=$(cygpath $CPYTHON_PATH)
        OPENBLAS_PATH=$(cygpath $OPENBLAS_PATH)
        export PATH="$PATH:$OPENBLAS_PATH/:$CPYTHON_PATH/"
        export PYTHONPATH="$INSTALL_PATH/lib/site-packages/"
        mkdir -p "$PYTHONPATH"
        # the build sometimes fails with multiple jobs
        MAKEJ=1
        # setup.py install doesn't accept absolute paths on Windows
        ATLAS=None "$CPYTHON_PATH/bin/python.exe" setup.py build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ -L$CPYTHON_PATH/libs/ install --prefix ..
        ;;
    windows-x86_64)
        CPYTHON_PATH=$(cygpath $CPYTHON_PATH)
        OPENBLAS_PATH=$(cygpath $OPENBLAS_PATH)
        export PATH="$PATH:$OPENBLAS_PATH/:$CPYTHON_PATH/"
        export PYTHONPATH="$INSTALL_PATH/lib/site-packages/"
        mkdir -p "$PYTHONPATH"
        # the build sometimes fails with multiple jobs
        MAKEJ=1
        # setup.py install doesn't accept absolute paths on Windows
        ATLAS=None "$CPYTHON_PATH/bin/python.exe" setup.py build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -L$CPYTHON_PATH/lib/ -L$CPYTHON_PATH/libs/ install --prefix ..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

ln -snf $PYTHONPATH/numpy-*/ ../python
rm -Rf $(find ../ -iname __pycache__)

cd ../..
