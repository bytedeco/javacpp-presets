#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libfreenect2
    popd
    exit
fi

LIBFREENECT2_VERSION=0.2.0
download https://github.com/OpenKinect/libfreenect2/archive/v$LIBFREENECT2_VERSION.zip libfreenect2-$LIBFREENECT2_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib bin
unzip -o ../libfreenect2-$LIBFREENECT2_VERSION.zip

cd libfreenect2-$LIBFREENECT2_VERSION

case $PLATFORM in
    linux-x86)
#	patch -Np1 < ../../../libfreenect2-$LIBFREENECT2_VERSION.patch
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_OPENNI_DRIVER=OFF -DENABLE_CUDA=OFF 	-DENABLE_CXX11=OFF -DENABLE_OPENCL=OFF -DENABLE_VAAPI=OFF -DENABLE_TEGRAJPEG=OFF -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
        ;;
    linux-x86_64)
#	patch -Np1 < ../../../libfreenect2-$LIBFREENECT2_VERSION.patch
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_OPENNI_DRIVER=OFF -DENABLE_CUDA=OFF 	-DENABLE_CXX11=OFF -DENABLE_OPENCL=OFF -DENABLE_VAAPI=OFF -DENABLE_TEGRAJPEG=OFF -DCMAKE_INSTALL_PREFIX=..
        make -j4
        make install
	patch -Np1 < ../../../libfreenect2-$LIBFREENECT2_VERSION.patch
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
