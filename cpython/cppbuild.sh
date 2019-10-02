#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" cpython
    popd
    exit
fi

CPYTHON_VERSION=3.7.4
download https://www.python.org/ftp/python/$CPYTHON_VERSION/Python-$CPYTHON_VERSION.tgz Python-$CPYTHON_VERSION.tgz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xzf ../Python-$CPYTHON_VERSION.tgz
cd Python-$CPYTHON_VERSION
patch -Np1 --binary < ../../../cpython-windows.patch

case $PLATFORM in
    linux-armhf)
        ./configure --prefix=$INSTALL_PATH/host --with-system-ffi
        make -j $MAKEJ
        make install
        make distclean
        export PATH=$INSTALL_PATH/host/bin/:$PATH
        CC="arm-linux-gnueabihf-gcc -std=c99 -march=armv6 -mfpu=vfp -mfloat-abi=hard" ./configure --prefix=$INSTALL_PATH --host=arm-linux-gnueabihf --build=$(uname -m)-pc-linux-gnu --enable-shared --with-system-ffi LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/' ac_cv_file__dev_ptmx=no ac_cv_file__dev_ptc=no --disable-ipv6
        make -j $MAKEJ
        make install
        ;;
    linux-arm64)
        ./configure --prefix=$INSTALL_PATH/host --with-system-ffi
        make -j $MAKEJ
        make install
        make distclean
        export PATH=$INSTALL_PATH/host/bin/:$PATH
        CC="aarch64-linux-gnu-gcc -mabi=lp64" ./configure --prefix=$INSTALL_PATH --host=aarch64-linux-gnu --build=$(uname -m)-pc-linux-gnu --enable-shared --with-system-ffi LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/' ac_cv_file__dev_ptmx=no ac_cv_file__dev_ptc=no --disable-ipv6
        make -j $MAKEJ
        make install
        ;;
    linux-ppc64le)
        ./configure --prefix=$INSTALL_PATH/host --with-system-ffi
        make -j $MAKEJ
        make install
        make distclean
        export PATH=$INSTALL_PATH/host/bin/:$PATH
        CC="powerpc64le-linux-gnu-gcc -m64" ./configure --prefix=$INSTALL_PATH --host=powerpc64le-linux-gnu --build=$(uname -m)-pc-linux-gnu --enable-shared --with-system-ffi LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/' ac_cv_file__dev_ptmx=no ac_cv_file__dev_ptc=no --disable-ipv6
        make -j $MAKEJ
        make install
        ;;
    linux-x86)
        CC="gcc -m32" ./configure --prefix=$INSTALL_PATH --enable-shared --with-system-ffi LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/'
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        CC="gcc -m64" ./configure --prefix=$INSTALL_PATH --enable-shared --with-system-ffi LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/'
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        ./configure --prefix=$INSTALL_PATH --enable-shared --with-openssl="$(brew --prefix openssl@1.1)" LDFLAGS='-s -Wl,-rpath,@loader_path/,-rpath,@loader_path/../,-rpath,@loader_path/../lib/'
        sedinplace 's:-install_name,$(prefix)/lib/:-install_name,@rpath/:g' Makefile
        make -j $MAKEJ
        make install
        ;;
    windows-x86)
        mkdir -p ../include ../lib ../libs ../bin
        cd PCbuild
        cmd.exe //c 'build.bat -p x86'
        cp win32/python*.exe win32/python*.dll ../../bin/
        cp win32/python*.lib ../../libs/
        cp win32/*.dll win32/*.pyd ../../lib/
        cd ..
        cp -r Lib/* ../lib/
        cp -r Include/* PC/pyconfig.h ../include/
        unzip -o ../lib/ensurepip/_bundled/pip* -d ../lib/
        unzip -o ../lib/ensurepip/_bundled/setuptools* -d ../lib/
        ;;
    windows-x86_64)
        mkdir -p ../include ../lib ../libs ../bin
        cd PCbuild
        cmd.exe //c 'build.bat -p x64'
        cp amd64/python*.exe amd64/python*.dll ../../bin/
        cp amd64/python*.lib ../../libs/
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
