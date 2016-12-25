#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libfreenect2
    popd
    exit
fi

LIBJPEG=libjpeg-turbo-1.5.1
LIBFREENECT2_VERSION=0.2.0
download http://downloads.sourceforge.net/project/libjpeg-turbo/1.5.1/$LIBJPEG.tar.gz $LIBJPEG.tar.gz
download https://github.com/OpenKinect/libfreenect2/archive/v$LIBFREENECT2_VERSION.zip libfreenect2-$LIBFREENECT2_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
tar -xzvf ../$LIBJPEG.tar.gz
mkdir -p include lib bin
unzip -o ../libfreenect2-$LIBFREENECT2_VERSION.zip

case $PLATFORM in
    linux-x86)
        export CC="$OLDCC -m32 -fPIC"
        cd $LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=i686-linux
        make -j $MAKEJ
        make install
        cd ../libfreenect2-$LIBFREENECT2_VERSION
#	patch -Np1 < ../../../libfreenect2-$LIBFREENECT2_VERSION.patch
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_OPENNI_DRIVER=OFF -DENABLE_CUDA=OFF -DENABLE_CXX11=OFF -DENABLE_OPENCL=OFF -DENABLE_VAAPI=OFF -DENABLE_TEGRAJPEG=OFF -DCMAKE_INSTALL_PREFIX=.. -DTurboJPEG_INCLUDE_DIRS=../include -DTurboJPEG_LIBRARIES=../lib/libturbojpeg.a
        make -j4
        make install
        patch -Np1 < ../../../libfreenect2-$LIBFREENECT2_VERSION.patch
        ;;
    linux-x86_64)
        export CC="$OLDCC -m64 -fPIC"
        cd $LIBJPEG
        ./configure --prefix=$INSTALL_PATH --disable-shared --with-pic --host=x86_64-linux
        make -j $MAKEJ
        make install
        cd ../libfreenect2-$LIBFREENECT2_VERSION
#	patch -Np1 < ../../../libfreenect2-$LIBFREENECT2_VERSION.patch
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_OPENNI_DRIVER=OFF -DENABLE_CUDA=OFF -DENABLE_CXX11=OFF -DENABLE_OPENCL=OFF -DENABLE_VAAPI=OFF -DENABLE_TEGRAJPEG=OFF -DCMAKE_INSTALL_PREFIX=.. -DTurboJPEG_INCLUDE_DIRS=../include -DTurboJPEG_LIBRARIES=../lib/libturbojpeg.a
        make -j4
        make install
        patch -Np1 < ../../../libfreenect2-$LIBFREENECT2_VERSION.patch
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
