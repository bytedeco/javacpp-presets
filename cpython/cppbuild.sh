#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" cpython
    popd
    exit
fi

CPYTHON_VERSION=3.7.3
download https://www.python.org/ftp/python/$CPYTHON_VERSION/Python-$CPYTHON_VERSION.tgz Python-$CPYTHON_VERSION.tgz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xzf ../Python-$CPYTHON_VERSION.tgz
cd Python-$CPYTHON_VERSION
patch -Np1 --binary < ../../../cpython-windows.patch

case $PLATFORM in
    linux-*)
        ./configure --prefix=$INSTALL_PATH --enable-shared --with-system-ffi LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../lib/'
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        ./configure --prefix=$INSTALL_PATH --enable-shared LDFLAGS='-s -Wl,-rpath,@loader_path/,-rpath,@loader_path/../lib/'
        sedinplace 's:-install_name,$(prefix)/lib/:-install_name,@rpath/:g' Makefile
        make -j $MAKEJ
        make install
        ;;
    windows-*)
        mkdir -p ../include ../lib ../bin
        cd PCbuild
        cmd.exe /c 'build.bat -p x64'
        cp amd64/python*.exe amd64/python*.dll ../../bin/
        cp amd64/python*.lib ../../lib/
        cp amd64/*.dll amd64/*.pyd ../../lib/
        cd ..
        cp -r Lib/* ../lib/
        cp -r Include/* PC/pyconfig.h ../include/
        unzip -o ../lib/ensurepip/_bundled/pip* -d ../lib/
        unzip -o ../lib/ensurepip/_bundled/setuptools* -d ../lib/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

rm -Rf $(find ../ -iname __pycache__)

cd ../..
