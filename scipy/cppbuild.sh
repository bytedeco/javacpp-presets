#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" scipy
    popd
    exit
fi

BOOST=1_75_0
HIGHS=1.1.1
SCIPY_VERSION=1.8.0
download http://downloads.sourceforge.net/project/boost/boost/${BOOST//_/.}/boost_$BOOST.tar.gz boost_$BOOST.tar.gz
download https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v$HIGHS.tar.gz HiGHS-$HIGHS.tar.gz
download https://github.com/scipy/unuran/archive/refs/heads/main.tar.gz unuran-main.tar.gz
download https://github.com/scipy/PROPACK/archive/refs/heads/main.tar.gz PROPACK-main.tar.gz
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
tar --totals -xzf ../boost_$BOOST.tar.gz
tar --totals -xzf ../HiGHS-$HIGHS.tar.gz
tar --totals -xzf ../unuran-main.tar.gz
tar --totals -xzf ../PROPACK-main.tar.gz
tar --totals -xzf ../scipy-$SCIPY_VERSION.tar.gz
cp -a boost_$BOOST/* scipy-$SCIPY_VERSION/scipy/_lib/boost/
#cp -a HiGHS-$HIGHS/* scipy-$SCIPY_VERSION/scipy/_lib/highs/
cp -a unuran-main/* scipy-$SCIPY_VERSION/scipy/_lib/unuran/
cp -a PROPACK-main/* scipy-$SCIPY_VERSION/scipy/sparse/linalg/_propack/PROPACK/
cd scipy-$SCIPY_VERSION

# prevent setuptools from trying to build NumPy
sedinplace '/req_np/d' setup.py

echo "[openblas]"                                  > site.cfg
echo "libraries = openblas"                       >> site.cfg
echo "library_dirs = $OPENBLAS_PATH/lib/"         >> site.cfg
echo "include_dirs = $OPENBLAS_PATH/include/"     >> site.cfg

if [[ -f "$CPYTHON_PATH/include/python3.10/Python.h" ]]; then
    # setup.py won't pick up the right libgfortran.so without this
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/:$NUMPY_PATH/lib/"
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

TOOLS="setuptools==59.1.0 cython==0.29.24 pybind11==2.6.2 pythran==0.10.0 decorator==5.1.0 six==1.16.0 networkx==2.6.3 ply==3.11 beniget==0.4.0 gast==0.5.0"
if ! $PYTHON_BIN_PATH -m pip install --no-deps --target=$PYTHON_LIB_PATH $TOOLS; then
    echo "extra_link_args = -lgfortran"           >> site.cfg
    chmod +x "$CPYTHON_HOST_PATH/bin/python3.10"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CPYTHON_HOST_PATH/lib/:$CPYTHON_HOST_PATH"
    "$CPYTHON_HOST_PATH/bin/python3.10" -m pip install --no-deps --target="$CPYTHON_HOST_PATH/lib/python3.10/" crossenv==1.0 numpy==1.22.2 $TOOLS
    "$CPYTHON_HOST_PATH/bin/python3.10" -m crossenv "$PYTHON_BIN_PATH" crossenv
    cp -a "$NUMPY_PATH/python/numpy" "$CPYTHON_HOST_PATH/lib/python3.10/"
#    cp -a "$CPYTHON_HOST_PATH/lib/python3.10/include" "$PYTHON_LIB_PATH"
    source crossenv/bin/activate
    cross-expose cython numpy pybind11 pythran
    chmod +x $CPYTHON_HOST_PATH/lib/python3.10/bin/*
    export PATH="$CPYTHON_HOST_PATH/lib/python3.10/bin/:$PATH"
    export PYTHON_BIN_PATH="python"
    export NUMPY_MADVISE_HUGEPAGE=1

    # For some reason, setup.py fails on Linux if the Python installation is not at its original prefix
    PREFIX_HOST_PATH=$(sed -n 's/^prefix="\(.*\)"/\1/p' $CPYTHON_HOST_PATH/bin/python3.10-config)
    mkdir -p $PREFIX_HOST_PATH
    cp -a $CPYTHON_HOST_PATH/* $PREFIX_HOST_PATH
fi

if [[ $PLATFORM == linux* ]]; then
    # For some reason, setup.py fails on Linux if the Python installation is not at its original prefix
    PREFIX_PATH=$(sed -n 's/^prefix="\(.*\)"/\1/p' $CPYTHON_PATH/bin/python3.10-config)
    mkdir -p $PREFIX_PATH
    cp -a $CPYTHON_PATH/* $PREFIX_PATH
fi

case $PLATFORM in
    linux-armhf)
        FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard"
        ATLAS=None CC="arm-linux-gnueabihf-gcc -std=c99 $FLAGS" F77="arm-linux-gnueabihf-gfortran" F90="$F77" FFLAGS="$FLAGS -fPIC" LDFLAGS="$FLAGS -shared" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas -lpthread -lgfortran install --prefix $INSTALL_PATH
        arm-linux-gnueabihf-strip $(find ../ -iname *.so)
        ;;
    linux-arm64)
        ATLAS=None CC="aarch64-linux-gnu-gcc -mabi=lp64" F77="aarch64-linux-gnu-gfortran" F90="$F77" FFLAGS="-mabi=lp64 -fPIC" LDFLAGS="-mabi=lp64 -shared" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas -lpthread -lgfortran install --prefix $INSTALL_PATH
        aarch64-linux-gnu-strip $(find ../ -iname *.so)
        ;;
    linux-ppc64le)
        ATLAS=None CC="powerpc64le-linux-gnu-gcc -m64" F77="powerpc64le-linux-gnu-gfortran" F90="$F77" FFLAGS="-m64 -fPIC" LDFLAGS="-m64 -shared" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas -lpthread -lgfortran install --prefix $INSTALL_PATH
        powerpc64le-linux-gnu-strip $(find ../ -iname *.so)
        ;;
    linux-x86)
        ATLAS=None CC="gcc -m32 -D__STDC_NO_THREADS__" FFLAGS="-m32 -fPIC" LDFLAGS="-m32 -shared" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas -lpthread -lgfortran install --prefix $INSTALL_PATH
        strip $(find ../ -iname *.so)
        ;;
    linux-x86_64)
        ATLAS=None CC="gcc -m64 -D__STDC_NO_THREADS__" FFLAGS="-m64 -fPIC" LDFLAGS="-m64 -shared" "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas -lpthread -lgfortran install --prefix $INSTALL_PATH
        strip $(find ../ -iname *.so)
        ;;
    macosx-*)
        export F77="$(ls -1 /usr/local/bin/gfortran-? | head -n 1)"
        export F90="$F77"
        export LDFLAGS="-L/usr/lib/"
        ATLAS=None "$PYTHON_BIN_PATH" setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$OPENBLAS_PATH/lib/ -lopenblas -lpthread -lgfortran install --prefix $INSTALL_PATH
        # need to add RPATH so it can find MKL in cache
        for f in $(find ../ -iname *.so); do install_name_tool -add_rpath @loader_path/../../../ -add_rpath @loader_path/../../../../ $f || true; done
        ;;
    windows-x86)
        # parameters required by clang-cl
        export CL="-m32"
        # the build sometimes fails with multiple jobs
        export MAKEJ=1
        # SciPy can only be built from very short paths on Windows
        cmd.exe //c "mklink /j \\scipy ."
        export CPYTHON_PATH=$(cygpath -w $CPYTHON_PATH)
        export PYTHON_BIN_PATH=$(cygpath -w $PYTHON_BIN_PATH)
        export PYTHON_LIB_PATH=$(cygpath -w $PYTHON_LIB_PATH)
        export OPENBLAS_PATH=$(cygpath -w $OPENBLAS_PATH)
        export INSTALL_PATH=$(cygpath -w $INSTALL_PATH)
        export ATLAS=None
        cmd.exe //c "cd \\scipy & $PYTHON_BIN_PATH setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$CPYTHON_PATH/libs/ -L$OPENBLAS_PATH/lib/ -lopenblas install --prefix $INSTALL_PATH"
        ;;
    windows-x86_64)
        # parameters required by clang-cl
        export CL="-m64"
        # the build sometimes fails with multiple jobs
        export MAKEJ=1
        # SciPy can only be built from very short paths on Windows
        cmd.exe //c "mklink /j \\scipy ."
        export CPYTHON_PATH=$(cygpath -w $CPYTHON_PATH)
        export PYTHON_BIN_PATH=$(cygpath -w $PYTHON_BIN_PATH)
        export PYTHON_LIB_PATH=$(cygpath -w $PYTHON_LIB_PATH)
        export OPENBLAS_PATH=$(cygpath -w $OPENBLAS_PATH)
        export INSTALL_PATH=$(cygpath -w $INSTALL_PATH)
        export ATLAS=None
        cmd.exe //c "cd \\scipy & $PYTHON_BIN_PATH setup.py --quiet build -j $MAKEJ build_ext -I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/ -L$CPYTHON_PATH/lib/ -L$CPYTHON_PATH/libs/ -L$OPENBLAS_PATH/lib/ -lopenblas install --prefix $INSTALL_PATH"
        cmd.exe //c "rmdir \\scipy"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

ln -snf $PYTHON_INSTALL_PATH/scipy-*/ ../python
rm -Rf $(find ../ -iname __pycache__)

cd ../..
