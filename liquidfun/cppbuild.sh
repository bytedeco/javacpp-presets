#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" liquidfun
    popd
    exit
fi

LIQUIDFUN_VERSION=43d53e063cd349f7d09ee9dd37842afcc0247f44 # 20170717
download https://github.com/google/liquidfun/archive/$LIQUIDFUN_VERSION.tar.gz liquidfun-$LIQUIDFUN_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
echo "Decompressing archives..."
tar --totals -xzf ../liquidfun-$LIQUIDFUN_VERSION.tar.gz
cd liquidfun-$LIQUIDFUN_VERSION

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
    windows-x86_64)
        cd liquidfun/Box2D
        CC="$OLDCC -m64" CXX="$OLDCXX -m64" CXXFLAGS="/Wv:17" $CMAKE -G "Visual Studio 14 2015 Win64" -DBOX2D_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBOX2D_BUILD_SHARED=ON -DBOX2D_BUILD_EXAMPLES=OFF -DBOX2D_BUILD_UNITTESTS=OFF
        MSBuild.exe Box2D.sln //p:Configuration=Release //maxcpucount:$MAKEJ
        ;;
    macosx-*)
        patch -Np1 <$INSTALL_PATH/../../liquidfun-macosx.patch
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
