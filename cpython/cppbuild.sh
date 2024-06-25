#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" cpython
    popd
    exit
fi

OPENSSL=openssl-3.3.0
CPYTHON_VERSION=3.12.4
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
        # armhf builds are no longer guaranteed to succeed.
        # If they are needed, look at the configure command for arm64
        # and try any options that are missing from here
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
        ./Configure $OS-$ARCH -fPIC no-shared --prefix=$INSTALL_PATH/host --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        make distclean
        if [ $CROSSCOMPILE -eq 1 ]
        then
          ./Configure linux-generic32 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib --cross-compile-prefix=arm-linux-gnueabihf-
        else
          ./Configure linux-generic32 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        fi
        make -s -j $MAKEJ
        make install_sw
        make distclean
        cd ../Python-$CPYTHON_VERSION
        # ac_cv_buggy_getaddrinfo=no disables the runtime ./configure checks for ipv6 support
        # Without it, the build fails because it can't detect ipv6 on the host. Needed on both host and cross-compiled builds
        ./configure --prefix=$INSTALL_PATH/host --with-system-ffi --with-openssl=$INSTALL_PATH/host ac_cv_buggy_getaddrinfo=no
        make -j $MAKEJ
        make install
        make distclean
        export PATH=$INSTALL_PATH/host/bin/:$PATH
        # ac_cv_file__dev_ptmx=yes and ac_cv_file__dev_ptc=no are required for cross-compilation as stated by the configure script,
        # but little information is known about them
        CC="arm-linux-gnueabihf-gcc -std=c99" ./configure --prefix=$INSTALL_PATH --host=arm-linux-gnueabihf --build=$(uname -m)-pc-linux-gnu --enable-shared --with-system-ffi --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/' --with-build-python=$INSTALL_PATH/host/bin/python3 $INSTALL_PATH/host ac_cv_working_openssl_hashlib=yes ac_cv_working_openssl_ssl=yes ac_cv_buggy_getaddrinfo=no ac_cv_file__dev_ptmx=yes ac_cv_file__dev_ptc=no
        make -j $MAKEJ
        make install
        ;;
    linux-arm64)
        CFLAGS="-march=armv8-a+crypto -mcpu=cortex-a57+crypto"
        cd ../$OPENSSL
        ./Configure $OS-$ARCH -fPIC no-shared --prefix=$INSTALL_PATH/host --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        make distclean
        ./Configure linux-aarch64 -fPIC --prefix=$INSTALL_PATH --libdir=lib --cross-compile-prefix=aarch64-linux-gnu- "$CFLAGS" no-shared
        make -s -j $MAKEJ
        make install_sw
        make distclean
        cd ../Python-$CPYTHON_VERSION
        # ac_cv_buggy_getaddrinfo=no disables the runtime ./configure checks for ipv6 support
        # Without it, the build fails because it can't detect ipv6 on the host. Needed on both host and cross-compiled builds
        ./configure --prefix=$INSTALL_PATH/host --with-system-ffi --with-openssl=$INSTALL_PATH/host ac_cv_buggy_getaddrinfo=no
        make -j $MAKEJ
        make install
        make distclean
        export PATH=$INSTALL_PATH/host/bin/:$PATH
        # ac_cv_file__dev_ptmx=yes and ac_cv_file__dev_ptc=no are required for cross-compilation as stated by the configure script,
        # but little information is known about them.
        # /dev/ptmx is the pseudoterminal master file, reading from it generates a new file descriptor
        # to use with a corresponding /dev/pts/ pseudoterminal
        # See man 4 ptmx
        # No information on /dev/ptc could be found.
        # The above configure options specify whether the corresponding device files
        # are expected to be found on the target machine.
        CC="aarch64-linux-gnu-gcc -mabi=lp64 $CFLAGS" ./configure --prefix=$INSTALL_PATH --host=aarch64-linux-gnu --build=$(uname -m)-pc-linux-gnu --enable-shared --with-system-ffi --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/' --with-build-python=$INSTALL_PATH/host/bin/python3 ac_cv_working_openssl_hashlib=yes ac_cv_working_openssl_ssl=yes ac_cv_buggy_getaddrinfo=no ac_cv_file__dev_ptmx=yes ac_cv_file__dev_ptc=no
        make -j $MAKEJ
        make install
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        cd ../$OPENSSL
        ./Configure $OS-$ARCH -fPIC no-shared --prefix=$INSTALL_PATH/host --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        make distclean
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./Configure linux-ppc64le -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        else
          ./Configure linux-ppc64le -fPIC no-shared --cross-compile-prefix=powerpc64le-linux-gnu- --prefix=$INSTALL_PATH --libdir=lib
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
        CC="powerpc64le-linux-gnu-gcc -m64" ./configure --prefix=$INSTALL_PATH --host=powerpc64le-linux-gnu --build=$(uname -m)-pc-linux-gnu --enable-shared --with-system-ffi --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/' --with-build-python=$INSTALL_PATH/host/bin/python3 ac_cv_working_openssl_hashlib=yes ac_cv_working_openssl_ssl=yes
        make -j $MAKEJ
        make install
        ;;
    linux-x86)
        cd ../$OPENSSL
        ./Configure linux-elf -m32 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../Python-$CPYTHON_VERSION
        CC="gcc -m32" ./configure --prefix=$INSTALL_PATH --enable-shared --with-system-ffi --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/' ac_cv_working_openssl_hashlib=yes ac_cv_working_openssl_ssl=yes
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        cd ../$OPENSSL
        ./Configure linux-x86_64 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../Python-$CPYTHON_VERSION
        CC="gcc -m64" ./configure --prefix=$INSTALL_PATH --enable-shared --with-system-ffi --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,\$$ORIGIN/,-rpath,\$$ORIGIN/../,-rpath,\$$ORIGIN/../lib/' ac_cv_working_openssl_hashlib=yes ac_cv_working_openssl_ssl=yes
        make -j $MAKEJ
        make install
        ;;
    macosx-arm64)
        cd ../$OPENSSL
        ./Configure darwin64-arm64 -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw

	# Without this variable, cpython will pick up openssl 1.1 from homebrew
	export PYTHON_BUILD_SKIP_HOMEBREW=1
        cd ../Python-$CPYTHON_VERSION
        sedinplace 's/libintl.h//g' configure
        sedinplace 's/ac_cv_lib_intl_textdomain=yes/ac_cv_lib_intl_textdomain=no/g' configure
        ./configure --prefix=$INSTALL_PATH --enable-shared --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,@loader_path/,-rpath,@loader_path/../,-rpath,@loader_path/../lib/' ac_cv_working_openssl_hashlib=yes ac_cv_working_openssl_ssl=yes
        sedinplace 's:-install_name,$(prefix)/lib/:-install_name,@rpath/:g' Makefile
        make -j $MAKEJ
        make install
        ;;
    macosx-x86_64)
        cd ../$OPENSSL
        ./Configure darwin64-x86_64-cc -fPIC no-shared --prefix=$INSTALL_PATH --libdir=lib
        make -s -j $MAKEJ
        make install_sw
        cd ../Python-$CPYTHON_VERSION
        sedinplace 's/libintl.h//g' configure
        sedinplace 's/ac_cv_lib_intl_textdomain=yes/ac_cv_lib_intl_textdomain=no/g' configure
        ./configure --prefix=$INSTALL_PATH --enable-shared --with-openssl=$INSTALL_PATH LDFLAGS='-s -Wl,-rpath,@loader_path/,-rpath,@loader_path/../,-rpath,@loader_path/../lib/' ac_cv_working_openssl_hashlib=yes ac_cv_working_openssl_ssl=yes
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
        # unzip -o ../lib/ensurepip/_bundled/setuptools* -d ../lib/
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
        # unzip -o ../lib/ensurepip/_bundled/setuptools* -d ../lib/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

rm -Rf $(find ../ -iname __pycache__)
#$PYTHON_BIN_PATH -m pip install --target=$PYTHON_LIB_PATH certifi --no-deps

cd ../..
