#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" liquidfun
    popd
    exit
fi

LIQUIDFUN_SHA=0708ce1 # 20150401
download https://github.com/google/liquidfun/archive/$LIQUIDFUN_SHA.zip liquidfun_$LIQUIDFUN_SHA.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
unzip -o ../liquidfun_$LIQUIDFUN_SHA.zip
cd `ls -d liquidfun-*`

case $PLATFORM in
    linux-x86)
        cd liquidfun/Box2D
        CC="$OLDCC -m32" CXX="$OLDCXX -m32" $CMAKE -DBOX2D_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBOX2D_BUILD_SHARED=ON -DBOX2D_BUILD_EXAMPLES=OFF -DBOX2D_BUILD_UNITTESTS=OFF
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        cd liquidfun/Box2D
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" $CMAKE -DBOX2D_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBOX2D_BUILD_SHARED=ON -DBOX2D_BUILD_EXAMPLES=OFF -DBOX2D_BUILD_UNITTESTS=OFF
        make -j $MAKEJ
        make install/strip
        ;;
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
