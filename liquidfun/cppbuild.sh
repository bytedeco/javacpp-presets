#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" liquidfun
    popd
    exit
fi

LIQUIDFUN_VERSION=1.1.0
LIQUIDFUN_SHA=0708ce1 # 2016-12-29
#download https://github.com/google/liquidfun/releases/download/v$LIQUIDFUN_VERSION/liquidfun-$LIQUIDFUN_VERSION.tar.gz liquidfun-$LIQUIDFUN_VERSION.tar.gz
download https://github.com/google/liquidfun/archive/$LIQUIDFUN_SHA.zip liquidfun_$LIQUIDFUN_SHA.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
#tar -xzvf ../liquidfun-$LIQUIDFUN_VERSION.tar.gz
unzip -o ../liquidfun_$LIQUIDFUN_SHA.zip
cd `ls -d liquidfun-*`

case $PLATFORM in
    macosx-*)
        patch -Np1 <$INSTALL_PATH/../../liquidfun-$LIQUIDFUN_SHA-macosx.patch
        cd liquidfun/Box2D
        $CMAKE -G "Xcode" -DBOX2D_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBOX2D_BUILD_SHARED=ON -DBOX2D_BUILD_EXAMPLES=OFF -DBOX2D_BUILD_UNITTESTS=OFF
        xcodebuild -project Box2D.xcodeproj -configuration Release -target install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd $INSTALL_PATH
cd ..
