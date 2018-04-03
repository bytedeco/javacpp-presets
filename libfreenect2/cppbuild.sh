#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libfreenect2
    popd
    exit
fi

LIBUSB_VERSION=1.0.21
GLFW_VERSION=3.2.1
LIBJPEG=libjpeg-turbo-1.5.3
LIBFREENECT2_VERSION=0.2.0
download http://sourceforge.net/projects/libusb/files/libusb-1.0/libusb-$LIBUSB_VERSION/libusb-$LIBUSB_VERSION.tar.bz2/download libusb-$LIBUSB_VERSION.tar.bz2
download https://github.com/glfw/glfw/archive/$GLFW_VERSION.tar.gz glfw-$GLFW_VERSION.tar.gz
download http://downloads.sourceforge.net/project/libjpeg-turbo/1.5.3/$LIBJPEG.tar.gz $LIBJPEG.tar.gz
download https://github.com/OpenKinect/libfreenect2/archive/v$LIBFREENECT2_VERSION.tar.gz libfreenect2-$LIBFREENECT2_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
mkdir -p include lib bin
tar --totals -xjf ../libusb-$LIBUSB_VERSION.tar.bz2
tar --totals -xzf ../glfw-$GLFW_VERSION.tar.gz
tar --totals -xzf ../$LIBJPEG.tar.gz
tar --totals -xzf ../libfreenect2-$LIBFREENECT2_VERSION.tar.gz

if [[ $PLATFORM == windows* ]]; then
    download https://github.com/OpenKinect/libfreenect2/releases/download/v$LIBFREENECT2_VERSION/libfreenect2-$LIBFREENECT2_VERSION-usbdk-vs2015-x64.zip libfreenect2-$LIBFREENECT2_VERSION-usbdk-vs2015-x64.zip

    unzip -o libfreenect2-$LIBFREENECT2_VERSION-usbdk-vs2015-x64.zip
fi

case $PLATFORM in
    linux-x86)
        export CC="$OLDCC -m32 -fPIC"
        cd libusb-$LIBUSB_VERSION
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../glfw-$GLFW_VERSION
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../libfreenect2-$LIBFREENECT2_VERSION
        patch -Np1 < ../../../libfreenect2.patch
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_OPENNI_DRIVER=OFF -DENABLE_CUDA=OFF -DENABLE_CXX11=OFF -DENABLE_OPENCL=OFF -DENABLE_VAAPI=OFF -DENABLE_TEGRAJPEG=OFF -DCMAKE_INSTALL_PREFIX=.. -DLibUSB_INCLUDE_DIRS=../include/libusb-1.0 -DLibUSB_LIBRARIES=../lib/libusb-1.0.a -DGLFW3_INCLUDE_DIRS=../include -DGLFW3_LIBRARY=../lib/libglfw3.a -DTurboJPEG_INCLUDE_DIRS=../include -DTurboJPEG_LIBRARIES=../lib/libturbojpeg.a -DCMAKE_SHARED_LINKER_FLAGS="-lX11 -lXrandr -lXinerama -lXxf86vm -lXcursor"
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        export CC="$OLDCC -m64 -fPIC"
        cd libusb-$LIBUSB_VERSION
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux --disable-udev
        make -j $MAKEJ
        make install
        cd ../glfw-$GLFW_VERSION
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ
        make install
        cd ../libfreenect2-$LIBFREENECT2_VERSION
        patch -Np1 < ../../../libfreenect2.patch
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_OPENNI_DRIVER=OFF -DENABLE_CUDA=OFF -DENABLE_CXX11=OFF -DENABLE_OPENCL=OFF -DENABLE_VAAPI=OFF -DENABLE_TEGRAJPEG=OFF -DCMAKE_INSTALL_PREFIX=.. -DLibUSB_INCLUDE_DIRS=../include/libusb-1.0 -DLibUSB_LIBRARIES=../lib/libusb-1.0.a -DGLFW3_INCLUDE_DIRS=../include -DGLFW3_LIBRARY=../lib/libglfw3.a -DTurboJPEG_INCLUDE_DIRS=../include -DTurboJPEG_LIBRARIES=../lib/libturbojpeg.a -DCMAKE_SHARED_LINKER_FLAGS="-lX11 -lXrandr -lXinerama -lXxf86vm -lXcursor"
        make -j $MAKEJ
        make install
        ;;
    macosx-x86_64)
        cd glfw-$GLFW_VERSION
        $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
        make -j $MAKEJ
        make install
        cd ../$LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic
        make -j $MAKEJ
        make install
        cd ../libfreenect2-$LIBFREENECT2_VERSION
        patch -Np1 < ../../../libfreenect2.patch
        LDFLAGS="-framework Cocoa -framework IOKit -framework CoreFoundation -framework CoreVideo" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_OPENNI_DRIVER=OFF -DENABLE_CUDA=OFF -DENABLE_CXX11=OFF -DENABLE_OPENCL=OFF -DENABLE_VAAPI=OFF -DENABLE_TEGRAJPEG=OFF -DCMAKE_INSTALL_PREFIX=.. -DLibUSB_INCLUDE_DIRS=/usr/local/include/libusb-1.0 -DLibUSB_LIBRARIES=/usr/local/lib/libusb-1.0.dylib -DGLFW3_INCLUDE_DIRS=../include -DGLFW3_LIBRARY=../lib/libglfw3.a -DTurboJPEG_INCLUDE_DIRS=../include -DTurboJPEG_LIBRARIES=../lib/libturbojpeg.a -DCMAKE_MACOSX_RPATH=ON
        make -j $MAKEJ
        make install
        ;;
    windows-x86_64)
        cp -a libfreenect2-$LIBFREENECT2_VERSION-usbdk-vs2015-x64/include/* include
        cp -a libfreenect2-$LIBFREENECT2_VERSION-usbdk-vs2015-x64/lib/* lib
        cp -a libfreenect2-$LIBFREENECT2_VERSION-usbdk-vs2015-x64/bin/* bin
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
