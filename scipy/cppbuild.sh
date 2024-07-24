#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" scipy
    popd
    exit
fi

BOOST=1_75_0
SCIPY_VERSION=1.14.0rc2
download http://downloads.sourceforge.net/project/boost/boost/${BOOST//_/.}/boost_$BOOST.tar.gz boost_$BOOST.tar.gz
download https://github.com/data-apis/array-api-compat/archive/fd22a73.tar.gz array-api-compat-fd22a73.tar.gz
download https://github.com/cobyqa/cobyqa/archive/7f40b6d.tar.gz cobyqa-7f40b6d.tar.gz
download https://github.com/scipy/HiGHS/archive/4a12295.tar.gz HiGHS-4a12295.tar.gz
download https://github.com/scipy/unuran/archive/21810c8.tar.gz unuran-21810c8.tar.gz
download https://github.com/scipy/pocketfft/archive/9367142.tar.gz pocketfft-9367142.tar.gz
download https://github.com/scipy/PROPACK/archive/96f6800.tar.gz PROPACK-96f6800.tar.gz
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

echo "Decompressing archives..."
tar --totals -xzf ../boost_$BOOST.tar.gz
tar --totals -xzf ../array-api-compat-*.tar.gz || true
tar --totals -xzf ../cobyqa-*.tar.gz
tar --totals -xzf ../HiGHS-*.tar.gz
tar --totals -xzf ../unuran-*.tar.gz
tar --totals -xzf ../pocketfft-*.tar.gz
tar --totals -xzf ../PROPACK-*.tar.gz
tar --totals -xzf ../scipy-$SCIPY_VERSION.tar.gz
cp -a boost_$BOOST/* scipy-$SCIPY_VERSION/scipy/_lib/boost_math/
cp -a array-api-compat-*/* scipy-$SCIPY_VERSION/scipy/_lib/array_api_compat/
cp -a cobyqa-*/* scipy-$SCIPY_VERSION/scipy/_lib/cobyqa/
cp -a HiGHS-*/* scipy-$SCIPY_VERSION/scipy/_lib/highs/
cp -a unuran-*/* scipy-$SCIPY_VERSION/scipy/_lib/unuran/
cp -a pocketfft-*/* scipy-$SCIPY_VERSION/scipy/_lib/pocketfft/
cp -a PROPACK-*/* scipy-$SCIPY_VERSION/scipy/sparse/linalg/_propack/PROPACK/
cd scipy-$SCIPY_VERSION

sedinplace "/blas = dependency(\['openblas', 'OpenBLAS'\])/c\\
includes = include_directories('$OPENBLAS_PATH/include/')\\
blas = declare_dependency(dependencies : cc.find_library('openblas', dirs: '$OPENBLAS_PATH/lib/'), include_directories : includes)\\
" scipy/meson.build

sedinplace "/lapack = dependency(\['openblas', 'OpenBLAS'\])/c\\
lapack = blas\\
" scipy/meson.build

mkdir -p scipy/_lib/boost_math/include
cp -a scipy/_lib/boost_math/boost scipy/_lib/boost_math/include
# mv _setup.py setup.py

# prevent setuptools from trying to build NumPy
# sedinplace '/req_np/d' setup.py
# sedinplace 's/README.rst/README.md/g' scipy/_lib/setup.py

echo "[openblas]"                                  > site.cfg
echo "libraries = openblas"                       >> site.cfg
echo "library_dirs = $OPENBLAS_PATH/lib/"         >> site.cfg
echo "include_dirs = $OPENBLAS_PATH/include/"     >> site.cfg

if [[ -f "$CPYTHON_PATH/include/python3.12/Python.h" ]]; then
    # setup.py won't pick up the right libgfortran.so without this
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/:$NUMPY_PATH/lib/"
    export PATH="$CPYTHON_PATH/lib/python3.12/bin/:$PATH"
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

# https://github.com/scipy/scipy/issues/15281
export SCIPY_USE_PYTHRAN=0

TOOLS="setuptools==67.6.1 cython==3.0.10"
if ! $PYTHON_BIN_PATH -m pip install --no-deps --target=$PYTHON_LIB_PATH $TOOLS; then
    echo "extra_link_args = -lgfortran"           >> site.cfg
    chmod +x "$CPYTHON_HOST_PATH/bin/python3.12"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CPYTHON_HOST_PATH/lib/:$CPYTHON_HOST_PATH"
    "$CPYTHON_HOST_PATH/bin/python3.12" -m pip install --no-deps --target="$CPYTHON_HOST_PATH/lib/python3.12/" crossenv==1.4 numpy==2.0.0 $TOOLS
    "$CPYTHON_HOST_PATH/bin/python3.12" -m crossenv "$PYTHON_BIN_PATH" crossenv
    cp -a "$NUMPY_PATH/python/numpy" "$CPYTHON_HOST_PATH/lib/python3.12/"
#    cp -a "$CPYTHON_HOST_PATH/lib/python3.12/include" "$PYTHON_LIB_PATH"
    source crossenv/bin/activate
    cross-expose cython numpy pybind11 pythran
    chmod +x $CPYTHON_HOST_PATH/lib/python3.12/bin/*
    export PATH="$CPYTHON_HOST_PATH/lib/python3.12/bin/:$PATH"
    export PYTHON_BIN_PATH="python"
    export NUMPY_MADVISE_HUGEPAGE=1

    # For some reason, setup.py fails on Linux if the Python installation is not at its original prefix
    PREFIX_HOST_PATH=$(sed -n 's/^prefix="\(.*\)"/\1/p' $CPYTHON_HOST_PATH/bin/python3.12-config)
    mkdir -p $PREFIX_HOST_PATH
    cp -a $CPYTHON_HOST_PATH/* $PREFIX_HOST_PATH
fi

if [[ $PLATFORM == linux* ]]; then
    # For some reason, setup.py fails on Linux if the Python installation is not at its original prefix
    PREFIX_PATH=$(sed -n 's/^prefix="\(.*\)"/\1/p' $CPYTHON_PATH/bin/python3.12-config)
    mkdir -p $PREFIX_PATH
    cp -a $CPYTHON_PATH/* $PREFIX_PATH
fi

case $PLATFORM in
    linux-armhf)
        ATLAS=None CC="arm-linux-gnueabihf-gcc -std=c99" F77="arm-linux-gnueabihf-gfortran" F90="$F77" FFLAGS="-fPIC" "$PYTHON_BIN_PATH" -m pip install . --prefix $INSTALL_PATH --config-settings=builddir=build
        arm-linux-gnueabihf-strip $(find ../ -iname *.so)
        ;;
    linux-arm64)
        ATLAS=None CC="aarch64-linux-gnu-gcc -mabi=lp64" F77="aarch64-linux-gnu-gfortran" F90="$F77" FFLAGS="-mabi=lp64 -fPIC" "$PYTHON_BIN_PATH" -m pip install . --prefix $INSTALL_PATH --config-settings=builddir=build
        aarch64-linux-gnu-strip $(find ../ -iname *.so)
        ;;
    linux-ppc64le)
        ATLAS=None CC="powerpc64le-linux-gnu-gcc -m64" F77="powerpc64le-linux-gnu-gfortran" F90="$F77" FFLAGS="-m64 -fPIC" "$PYTHON_BIN_PATH" -m pip install . --prefix $INSTALL_PATH --config-settings=builddir=build
        powerpc64le-linux-gnu-strip $(find ../ -iname *.so)
        ;;
    linux-x86)
        ATLAS=None CC="gcc -m32 -D__STDC_NO_THREADS__" FFLAGS="-m32 -fPIC" "$PYTHON_BIN_PATH" -m pip install . --prefix $INSTALL_PATH --config-settings=builddir=build
        strip $(find ../ -iname *.so)
        ;;
    linux-x86_64)
        ATLAS=None CC="gcc -m64 -D__STDC_NO_THREADS__" FFLAGS="-m64 -fPIC" "$PYTHON_BIN_PATH" -m pip install . --prefix $INSTALL_PATH --config-settings=builddir=build
        strip $(find ../ -iname *.so)
        ;;
    macosx-arm64)
        export F77=$(compgen -cX '!gfortran*')
        export F90="$F77"
        export LDFLAGS="-L/usr/lib/"
        ATLAS=None FC="$F77" "$PYTHON_BIN_PATH" -m pip install . --prefix $INSTALL_PATH --config-settings=builddir=build
        # need to add RPATH so it can find MKL in cache
        for f in $(find ../ -iname *.so); do
          install_name_tool -add_rpath @loader_path/../../../ -add_rpath @loader_path/../../../../ $f || true;
          codesign --force -s - $f
        done
        ;;
    macosx-x86_64)
        export F77="$(ls -1 /usr/local/bin/gfortran-* | head -n 1)"
        export F90="$F77"
        export LDFLAGS="-L/usr/lib/"
        ATLAS=None FC="$F77" "$PYTHON_BIN_PATH" -m pip install . --prefix $INSTALL_PATH --config-settings=builddir=build
        # need to add RPATH so it can find MKL in cache
        for f in $(find ../ -iname *.so); do install_name_tool -add_rpath @loader_path/../../../ -add_rpath @loader_path/../../../../ $f || true; done
        ;;
    windows-x86)
        # parameters required by clang-cl
        export CL="-m32"
        ATLAS=None CC="gcc -m32" CXX="g++ -m32" "$PYTHON_BIN_PATH" -m pip install . --prefix $INSTALL_PATH --config-settings=builddir=build
        ;;
    windows-x86_64)
        # parameters required by clang-cl
        export CL="-m64"
        ATLAS=None CC="gcc -m64" CXX="g++ -m64" "$PYTHON_BIN_PATH" -m pip install . --prefix $INSTALL_PATH --config-settings=builddir=build
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cat build/meson-logs/meson-log.txt

if [[ -d $PYTHON_INSTALL_PATH/scipy ]]; then
    ln -snf $PYTHON_INSTALL_PATH ../python
else
    ln -snf $PYTHON_INSTALL_PATH/scipy-*/ ../python
fi
rm -Rf $(find ../ -iname __pycache__)

cd ../..
