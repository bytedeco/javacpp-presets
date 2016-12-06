#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libfreenect
    popd
    exit
fi

LIBFREENECT_VERSION=0.5.3
download https://github.com/OpenKinect/libfreenect/archive/v$LIBFREENECT_VERSION.zip libfreenect-$LIBFREENECT_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
unzip -o ../libfreenect-$LIBFREENECT_VERSION.zip

if [[ $PLATFORM == windows* ]]; then
    download http://downloads.sourceforge.net/project/libusb-win32/libusb-win32-releases/1.2.6.0/libusb-win32-bin-1.2.6.0.zip libusb-win32-bin-1.2.6.0.zip
    download ftp://sourceware.org/pub/pthreads-win32/pthreads-w32-2-9-1-release.zip pthreads-w32-2-9-1-release.zip

    unzip -o libusb-win32-bin-1.2.6.0.zip
    unzip -o pthreads-w32-2-9-1-release.zip -d pthreads-w32-2-9-1-release/
    patch -Np1 -d libfreenect-$LIBFREENECT_VERSION < ../../libfreenect-$LIBFREENECT_VERSION-windows.patch
fi

if [[ $PLATFORM == linux-armhf ]]; then
    download http://sourceforge.net/projects/libusb/files/libusb-1.0/libusb-1.0.19/libusb-1.0.19.tar.bz2/download libusb-1.0.19.tar.bz2
    tar xvjf libusb-1.0.19.tar.bz2
    cd libusb-1.0.19
    CFLAGS="-march=armv6 -marm -mfpu=vfp -mfloat-abi=hard" CXXFLAGS="-march=armv6 -marm -mfpu=vfp -mfloat-abi=hard" CPPFLAGS="-march=armv6 -marm -mfpu=vfp -mfloat-abi=hard" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=arm-linux-gnueabihf --disable-udev
    make
    make install
    cd .. 
fi

cd libfreenect-$LIBFREENECT_VERSION

case $PLATFORM in
    linux-x86)
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    linux-x86_64)
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    linux-armhf)
        CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ $CMAKE -DENABLE_SHARED=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=.. -DLIBUSB_1_INCLUDE_DIR=$INSTALL_PATH/include/libusb-1.0/ -DLIBUSB_1_LIBRARY=$INSTALL_PATH/lib/
        make -j4
        make install
        ;;
    linux-ppc64le)
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    macosx-*)
        patch -Np1 < ../../../libfreenect-$LIBFREENECT_VERSION-macosx.patch
        $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    windows-x86)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DLIBUSB_1_INCLUDE_DIR="../libusb-win32-bin-1.2.6.0/include" -DLIBUSB_1_LIBRARY="../libusb-win32-bin-1.2.6.0/lib/msvc/libusb.lib" -DTHREADS_PTHREADS_INCLUDE_DIR="../pthreads-w32-2-9-1-release/Pre-built.2/include" -DTHREADS_PTHREADS_WIN32_LIBRARY="../pthreads-w32-2-9-1-release/Pre-built.2/lib/x86/pthreadVC2.lib" -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=..
        nmake
        nmake install
        cp -r ../libusb-win32-bin-1.2.6.0/lib/msvc/* ../lib
        cp -r ../libusb-win32-bin-1.2.6.0/bin/x86/* ../bin
        cp -r ../pthreads-w32-2-9-1-release/Pre-built.2/lib/x86/* ../lib
        cp -r ../pthreads-w32-2-9-1-release/Pre-built.2/dll/x86/* ../bin
        ;;
    windows-x86_64)
        "$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DLIBUSB_1_INCLUDE_DIR="../libusb-win32-bin-1.2.6.0/include" -DLIBUSB_1_LIBRARY="../libusb-win32-bin-1.2.6.0/lib/msvc_x64/libusb.lib" -DTHREADS_PTHREADS_INCLUDE_DIR="../pthreads-w32-2-9-1-release/Pre-built.2/include" -DTHREADS_PTHREADS_WIN32_LIBRARY="../pthreads-w32-2-9-1-release/Pre-built.2/lib/x64/pthreadVC2.lib" -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DCMAKE_INSTALL_PREFIX=..
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
