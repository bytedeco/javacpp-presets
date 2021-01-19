#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libfreenect
    popd
    exit
fi

LIBFREENECT_VERSION=0.5.7
LIBUSB_VERSION=1.0.22
download https://github.com/OpenKinect/libfreenect/archive/v$LIBFREENECT_VERSION.tar.gz libfreenect-$LIBFREENECT_VERSION.tar.gz
download http://sourceforge.net/projects/libusb/files/libusb-1.0/libusb-$LIBUSB_VERSION/libusb-$LIBUSB_VERSION.tar.bz2/download libusb-$LIBUSB_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
mkdir -p include lib bin
tar --totals -xzf ../libfreenect-$LIBFREENECT_VERSION.tar.gz
tar --totals -xjf ../libusb-$LIBUSB_VERSION.tar.bz2

if [[ $PLATFORM == windows* ]]; then
    download http://downloads.sourceforge.net/project/libusb-win32/libusb-win32-releases/1.2.6.0/libusb-win32-bin-1.2.6.0.zip libusb-win32-bin-1.2.6.0.zip
    download http://sourceware.org/pub/pthreads-win32/pthreads-w32-2-9-1-release.zip pthreads-w32-2-9-1-release.zip

    unzip -o libusb-win32-bin-1.2.6.0.zip
    unzip -o pthreads-w32-2-9-1-release.zip -d pthreads-w32-2-9-1-release/
    patch -Np1 -d libfreenect-$LIBFREENECT_VERSION < ../../libfreenect-windows.patch
fi

cd libfreenect-$LIBFREENECT_VERSION

case $PLATFORM in
    linux-x86)
        cd ../libusb-$LIBUSB_VERSION
        CC="gcc -m32" CXX="g++ -m32" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../libfreenect-$LIBFREENECT_VERSION
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=.. -DLIBUSB_1_INCLUDE_DIR=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB_1_LIBRARY=$INSTALL_PATH/lib/libusb-1.0.a .
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        cd ../libusb-$LIBUSB_VERSION
        CC="gcc -m64" CXX="g++ -m64" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../libfreenect-$LIBFREENECT_VERSION
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=.. -DLIBUSB_1_INCLUDE_DIR=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB_1_LIBRARY=$INSTALL_PATH/lib/libusb-1.0.a .
        make -j $MAKEJ
        make install
        ;;
    linux-armhf)
        cd ../libusb-$LIBUSB_VERSION
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf --disable-udev
        make -j $MAKEJ
        make install
        cd ../libfreenect-$LIBFREENECT_VERSION
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=.. -DLIBUSB_1_INCLUDE_DIR=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB_1_LIBRARY=$INSTALL_PATH/lib/libusb-1.0.a .
        make -j $MAKEJ
        make install
        ;;
    linux-arm64)
        cd ../libusb-$LIBUSB_VERSION
        CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ CFLAGS="-march=armv8-a -mcpu=cortex-a57" CXXFLAGS="-march=armv8-a -mcpu=cortex-a57" CPPFLAGS="-march=armv8-a -mcpu=cortex-a57" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=aarch64-linux-gnu --disable-udev
        make -j $MAKEJ
        make install
        cd ../libfreenect-$LIBFREENECT_VERSION
        CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=.. -DLIBUSB_1_INCLUDE_DIR=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB_1_LIBRARY=$INSTALL_PATH/lib/libusb-1.0.a .
        make -j $MAKEJ
        make install
        ;;
    linux-ppc64le)
        cd ../libusb-$LIBUSB_VERSION
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
            CC="gcc -m64" CXX="g++ -m64" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=powerpc-linux-gnu --disable-udev
        else
            CC=powerpc64le-linux-gnu-gcc CXX=powerpc64le-linux-gnu-g++ ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=powerpc-linux-gnu --disable-udev
        fi
        make -j $MAKEJ
        make install
        cd ../libfreenect-$LIBFREENECT_VERSION
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          CC="gcc -m64" CXX="g++ -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=.. -DLIBUSB_1_INCLUDE_DIR=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB_1_LIBRARY=$INSTALL_PATH/lib/libusb-1.0.a .
        else
          CC=powerpc64le-linux-gnu-gcc CXX=powerpc64le-linux-gnu-g++ CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX $CMAKE -DCMAKE_SYSTEM_PROCESSOR=powerpc -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=.. -DLIBUSB_1_INCLUDE_DIR=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB_1_LIBRARY=$INSTALL_PATH/lib/libusb-1.0.a .
        fi
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        patch -Np1 < ../../../libfreenect-macosx.patch
        $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=.. .
        make -j $MAKEJ
        make install
        install_name_tool -change /usr/local/opt/libusb/lib/libusb-1.0.0.dylib @rpath/libusb-1.0.0.dylib ../lib/libfreenect.dylib
        install_name_tool -change /usr/local/opt/libusb/lib/libusb-1.0.0.dylib @rpath/libusb-1.0.0.dylib ../lib/libfreenect_sync.dylib
        ;;
    windows-x86)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DLIBUSB_1_INCLUDE_DIR="../libusb-win32-bin-1.2.6.0/include" -DLIBUSB_1_LIBRARY="../libusb-win32-bin-1.2.6.0/lib/msvc/libusb.lib" -DTHREADS_PTHREADS_INCLUDE_DIR="../pthreads-w32-2-9-1-release/Pre-built.2/include" -DTHREADS_PTHREADS_WIN32_LIBRARY="../pthreads-w32-2-9-1-release/Pre-built.2/lib/x86/pthreadVC2.lib" -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=.. .
        nmake
        nmake install
        cp -r ../libusb-win32-bin-1.2.6.0/lib/msvc/* ../lib
        cp -r ../libusb-win32-bin-1.2.6.0/bin/x86/* ../bin
        cp -r ../pthreads-w32-2-9-1-release/Pre-built.2/lib/x86/* ../lib
        cp -r ../pthreads-w32-2-9-1-release/Pre-built.2/dll/x86/* ../bin
        ;;
    windows-x86_64)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DLIBUSB_1_INCLUDE_DIR="../libusb-win32-bin-1.2.6.0/include" -DLIBUSB_1_LIBRARY="../libusb-win32-bin-1.2.6.0/lib/msvc_x64/libusb.lib" -DTHREADS_PTHREADS_INCLUDE_DIR="../pthreads-w32-2-9-1-release/Pre-built.2/include" -DTHREADS_PTHREADS_WIN32_LIBRARY="../pthreads-w32-2-9-1-release/Pre-built.2/lib/x64/pthreadVC2.lib" -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=.. .
        nmake
        nmake install
        cp -r ../libusb-win32-bin-1.2.6.0/lib/msvc_x64/* ../lib
        cp -r ../libusb-win32-bin-1.2.6.0/bin/amd64/* ../bin
        cp -r ../pthreads-w32-2-9-1-release/Pre-built.2/lib/x64/* ../lib
        cp -r ../pthreads-w32-2-9-1-release/Pre-built.2/dll/x64/* ../bin
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
