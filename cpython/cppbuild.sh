#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" cpython
    popd
    exit
fi

OPENSSL=openssl-1.1.1m
CPYTHON_VERSION=3.10.2
download https://www.openssl.org/source/$OPENSSL.tar.gz $OPENSSL.tar.gz
download https://www.python.org/ftp/python/$CPYTHON_VERSION/Python-$CPYTHON_VERSION.tgz Python-$CPYTHON_VERSION.tgz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
tar --totals -xzf ../$OPENSSL.tar.gz
tar --totals -xzf ../Python-$CPYTHON_VERSION.tgz
cd Python-$CPYTHON_VERSION
patch -Np1 --binary < ../../../cpython-windows.patch

case $PLATFORM in
    linux-armhf)
        HOST_ARCH="$(uname -m)"
        CROSSCOMPILE=1
        if [[ $HOST_ARCH == *"arm"* ]]
        then
          echo "Detected arm arch so not cross compiling";
          CROSSCOMPILE=0
        else
          echo "Detected non arm arch so cross compiling";
        fi

        cd ../$OPENSSL
        ./Configure $OS-$ARCH -fPIC no-shared --prefix=$INSTALL_PATH/host
        make -s -j $MAKEJ
        make install_sw
        make distclean
        if [ $CROSSCOMPILE -eq 1 ]
        then
          ./Configure linux-generic32 -march=armv6 -mfpu=vfp -mfloat-abi=hard -fPIC no-shared --prefix=$INSTALL_PATH --cross-compile-prefix=arm-linux-gnueabihf-
        else
          ./Configure linux-generic32 -fPIC no-shared --prefix=$INSTALL_PATH
        fi
        make -s -j $MAKEJ
        make install_sw
        make distclean
        cd ../Python-$CPYTHON_VERSION
        ./configure --prefix=$INSTALL_PATH/host --with-system-ffi --with-openssl=$INSTALL_PATH/host
        make -j $MAKEJ
        make install
        make distclean
        export PATH=$INSTALL_PATH/host/bin/:$PATH
        CC="arm-linux-gnueabihf-gcc -std=c99 -march=armv6 -mfpu=vfp -mfloat-abi=hard" ./configure --prefix=$INSTALL_PATH --host=arm-linux-gnueabihf --build=$(uname -m)-pc-linux-gnu --enable-shared --with-system-ffi --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/' ac_cv_file__dev_ptmx=no ac_cv_file__dev_ptc=no --disable-ipv6
        make -j $MAKEJ
        make install
        ;;
    linux-arm64)
        CFLAGS="-march=armv8-a+crypto -mcpu=cortex-a57+crypto"
        cd ../$OPENSSL
        ./Configure $OS-$ARCH -fPIC no-shared --prefix=$INSTALL_PATH/host
        make -s -j $MAKEJ
        make install_sw
        make distclean
        ./Configure linux-aarch64 -fPIC --prefix=$INSTALL_PATH --cross-compile-prefix=aarch64-linux-gnu- "$CFLAGS" no-shared
        make -s -j $MAKEJ
        make install_sw
        make distclean
        cd ../Python-$CPYTHON_VERSION
        ./configure --prefix=$INSTALL_PATH/host --with-system-ffi --with-openssl=$INSTALL_PATH/host
        make -j $MAKEJ
        make install
        make distclean
        export PATH=$INSTALL_PATH/host/bin/:$PATH
        CC="aarch64-linux-gnu-gcc -mabi=lp64 $CFLAGS" ./configure --prefix=$INSTALL_PATH --host=aarch64-linux-gnu --build=$(uname -m)-pc-linux-gnu --enable-shared --with-system-ffi --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/' ac_cv_file__dev_ptmx=no ac_cv_file__dev_ptc=no --disable-ipv6
        make -j $MAKEJ
        make install
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        cd ../$OPENSSL
        ./Configure $OS-$ARCH -fPIC no-shared --prefix=$INSTALL_PATH/host
        make -s -j $MAKEJ
        make install_sw
        make distclean
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./Configure linux-ppc64le -fPIC no-shared --prefix=$INSTALL_PATH
        else
          ./Configure linux-ppc64le -fPIC no-shared --cross-compile-prefix=powerpc64le-linux-gnu- --prefix=$INSTALL_PATH
        fi
        make -s -j $MAKEJ
        make install_sw
        make distclean
        cd ../Python-$CPYTHON_VERSION
        ./configure --prefix=$INSTALL_PATH/host --with-system-ffi --with-openssl=$INSTALL_PATH/host
        make -j $MAKEJ
        make install
        make distclean
        export PATH=$INSTALL_PATH/host/bin/:$PATH
        CC="powerpc64le-linux-gnu-gcc -m64" ./configure --prefix=$INSTALL_PATH --host=powerpc64le-linux-gnu --build=$(uname -m)-pc-linux-gnu --enable-shared --with-system-ffi --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/' ac_cv_file__dev_ptmx=no ac_cv_file__dev_ptc=no --disable-ipv6
        make -j $MAKEJ
        make install
        ;;
    linux-x86)
        cd ../$OPENSSL
        ./Configure linux-elf -m32 -fPIC no-shared --prefix=$INSTALL_PATH
        make -s -j $MAKEJ
        make install_sw
        cd ../Python-$CPYTHON_VERSION
        CC="gcc -m32" ./configure --prefix=$INSTALL_PATH --enable-shared --with-system-ffi --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/'
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        cd ../$OPENSSL
        ./Configure linux-x86_64 -fPIC no-shared --prefix=$INSTALL_PATH
        make -s -j $MAKEJ
        make install_sw
        cd ../Python-$CPYTHON_VERSION
        CC="gcc -m64" ./configure --prefix=$INSTALL_PATH --enable-shared --with-system-ffi --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/'
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        cd ../$OPENSSL
        ./Configure darwin64-x86_64-cc -fPIC no-shared --prefix=$INSTALL_PATH
        make -s -j $MAKEJ
        make install_sw
        cd ../Python-$CPYTHON_VERSION
        sedinplace 's/libintl.h//g' configure
        sedinplace 's/ac_cv_lib_intl_textdomain=yes/ac_cv_lib_intl_textdomain=no/g' configure
        ./configure --prefix=$INSTALL_PATH --enable-shared --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,@loader_path/,-rpath,@loader_path/../,-rpath,@loader_path/../lib/'
        sedinplace 's:-install_name,$(prefix)/lib/:-install_name,@rpath/:g' Makefile
        make -j $MAKEJ
        make install
        ;;
    windows-x86)
        mkdir -p ../include ../lib ../libs ../bin
        cd PCbuild
        cmd.exe //c 'build.bat -p x86 -vv'
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
        cmd.exe //c 'build.bat -p x64 -vv'
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
#$PYTHON_BIN_PATH -m pip install --target=$PYTHON_LIB_PATH certifi --no-deps

cd ../..
