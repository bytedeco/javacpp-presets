#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" scipy
    popd
    exit
fi

SCIPY_VERSION=1.6.0
download https://github.com/scipy/scipy/archive/v$SCIPY_VERSION.tar.gz scipy-$SCIPY_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

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
        elif [[ -f "$P/python/numpy/core/include/numpy/numpyconfig.h" ]]; then
            NUMPY_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

CPYTHON_HOST_PATH="${CPYTHON_HOST_PATH//\\//}"
CPYTHON_PATH="${CPYTHON_PATH//\\//}"
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"
NUMPY_PATH="${NUMPY_PATH//\\//}"

echo "Decompressing archives..."
tar --totals -xzf ../scipy-$SCIPY_VERSION.tar.gz
cd scipy-$SCIPY_VERSION

echo "[openblas]"                                  > site.cfg
echo "libraries = openblas"                       >> site.cfg
echo "library_dirs = $OPENBLAS_PATH/lib/"         >> site.cfg
echo "include_dirs = $OPENBLAS_PATH/include/"     >> site.cfg

if [[ -f "$CPYTHON_PATH/include/python3.8/Python.h" ]]; then
    # setup.py won't pick up the right libgfortran.so without this
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/:$NUMPY_PATH/lib/"
    export PYTHON_BIN_PATH="$CPYTHON_PATH/bin/python3.8"
    export PYTHON_INCLUDE_PATH="$CPYTHON_PATH/include/python3.8/"
    export PYTHON_LIB_PATH="$CPYTHON_PATH/lib/python3.8/"
    export PYTHON_INSTALL_PATH="$INSTALL_PATH/lib/python3.8/site-packages/"
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
fi
export PYTHONPATH="$PYTHON_INSTALL_PATH:$NUMPY_PATH/python/"
mkdir -p "$PYTHON_INSTALL_PATH"

if ! $PYTHON_BIN_PATH -m pip install --target=$PYTHON_LIB_PATH cython pybind11; then
    echo "extra_link_args = -lgfortran"           >> site.cfg
    chmod +x "$CPYTHON_HOST_PATH/bin/python3.8"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CPYTHON_HOST_PATH/lib/:$CPYTHON_HOST_PATH"
    "$CPYTHON_HOST_PATH/bin/python3.8" -m pip install --target="$CPYTHON_HOST_PATH/lib/python3.8/" crossenv cython numpy pybind11
    "$CPYTHON_HOST_PATH/bin/python3.8" -m crossenv "$PYTHON_BIN_PATH" crossenv
    cp "$NUMPY_PATH/python/numpy/core/lib/libnpymath.a" "$CPYTHON_HOST_PATH/lib/python3.8/numpy/core/lib/libnpymath.a"
#    cp -a "$CPYTHON_HOST_PATH/lib/python3.8/include" "$PYTHON_LIB_PATH"
    source crossenv/bin/activate
    cross-expose cython numpy pybind11
    PYTHON_BIN_PATH="python"
    export NUMPY_MADVISE_HUGEPAGE=1

    # For some reason, setup.py fails on Linux if the Python installation is not at its original prefix
    PREFIX_HOST_PATH=$(sed -n 's/^prefix="\(.*\)"/\1/p' $CPYTHON_HOST_PATH/bin/python3.8-config)
    mkdir -p $PREFIX_HOST_PATH
    cp -a $CPYTHON_HOST_PATH/* $PREFIX_HOST_PATH
fi

if [[ $PLATFORM == linux* ]]; then
    # For some reason, setup.py fails on Linux if the Python installation is not at its original prefix
    PREFIX_PATH=$(sed -n 's/^prefix="\(.*\)"/\1/p' $CPYTHON_PATH/bin/python3.8-config)
    mkdir -p $PREFIX_PATH
    cp -a $CPYTHON_PATH/* $PREFIX_PATH
fi

case $PLATFORM in
    linux-armhf)
        FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard"
        ATLAS=None CC="arm-linux-gnueabihf-gcc -std=c99 $FLAGS" F90="arm-linux-gnueabihf-gfortran" FFLAGS="$FLAGS -fPIC" LDFLAGS="$FLAGS -shared" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas -lpthread install --prefix $INSTALL_PATH
        arm-linux-gnueabihf-strip $(find ../ -iname *.so)
        ;;
    linux-arm64)
        ATLAS=None CC="aarch64-linux-gnu-gcc -mabi=lp64" F90="aarch64-linux-gnu-gfortran" FFLAGS="-mabi=lp64 -fPIC" LDFLAGS="-mabi=lp64 -shared" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas -lpthread install --prefix $INSTALL_PATH
        aarch64-linux-gnu-strip $(find ../ -iname *.so)
        ;;
    linux-ppc64le)
        ATLAS=None CC="powerpc64le-linux-gnu-gcc -m64" F90="powerpc64le-linux-gnu-gfortran" FFLAGS="-m64 -fPIC" LDFLAGS="-m64 -shared" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas -lpthread install --prefix $INSTALL_PATH
        powerpc64le-linux-gnu-strip $(find ../ -iname *.so)
        ;;
    linux-x86)
        ATLAS=None CC="gcc -m32" FFLAGS="-m32 -fPIC" LDFLAGS="-m32 -shared" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas install --prefix $INSTALL_PATH
        strip $(find ../ -iname *.so)
        ;;
    linux-x86_64)
        ATLAS=None CC="gcc -m64" FFLAGS="-m64 -fPIC" LDFLAGS="-m64 -shared" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas install --prefix $INSTALL_PATH
        strip $(find ../ -iname *.so)
        ;;
    macosx-*)
        export F90="$(ls -1 /usr/local/bin/gfortran-? | head -n 1)"
        ATLAS=None "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas install --prefix $INSTALL_PATH
        # need to add RPATH so it can find MKL in cache
        for f in $(find ../ -iname *.so); do install_name_tool -add_rpath @loader_path/../../../ -add_rpath @loader_path/../../../../ $f || true; done
        ;;
    windows-x86)
        # the build sometimes fails with multiple jobs
        MAKEJ=1
        # setup.py install doesn't accept absolute paths on Windows
        ATLAS=None "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$CPYTHON_PATH/libs/ -L$OPENBLAS_PATH/lib/ -lopenblas install --prefix ..
        ;;
    windows-x86_64)
        # the build sometimes fails with multiple jobs
        MAKEJ=1
        # setup.py install doesn't accept absolute paths on Windows
        ATLAS=None "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$CPYTHON_PATH/libs/ -L$OPENBLAS_PATH/lib/ -lopenblas install --prefix ..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

ln -snf $PYTHON_INSTALL_PATH/scipy-*/ ../python
rm -Rf $(find ../ -iname __pycache__)

cd ../..
