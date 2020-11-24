#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" liquidfun
    popd
    exit
fi

LIQUIDFUN_VERSION=master
download https://github.com/google/liquidfun/archive/$LIQUIDFUN_VERSION.tar.gz liquidfun-$LIQUIDFUN_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
echo "Decompressing archives..."
tar --totals -xzf ../liquidfun-$LIQUIDFUN_VERSION.tar.gz
cd liquidfun-$LIQUIDFUN_VERSION
sedinplace /WX/d liquidfun/Box2D/CMakeLists.txt
sedinplace s/-Werror//g liquidfun/Box2D/CMakeLists.txt

case $PLATFORM in
    linux-x86)
        cd liquidfun/Box2D
        CC="gcc -m32" CXX="g++ -m32" $CMAKE -DBOX2D_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBOX2D_BUILD_SHARED=ON -DBOX2D_BUILD_EXAMPLES=OFF -DBOX2D_BUILD_UNITTESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    linux-x86_64)
        cd liquidfun/Box2D
        CC="gcc -m64" CXX="g++ -m64" $CMAKE -DBOX2D_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBOX2D_BUILD_SHARED=ON -DBOX2D_BUILD_EXAMPLES=OFF -DBOX2D_BUILD_UNITTESTS=OFF .
        make -j $MAKEJ
        make install/strip
        ;;
    windows-x86)
        cd liquidfun/Box2D
        export CC="cl.exe"
        export CXX="cl.exe"
        CXXFLAGS="/Wv:17" $CMAKE -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DBOX2D_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBOX2D_BUILD_EXAMPLES=OFF -DBOX2D_BUILD_UNITTESTS=OFF .
        ninja -j $MAKEJ
        cp -r Box2D $INSTALL_PATH/include
        cp Box2D/*.lib $INSTALL_PATH/lib
        ;;
    windows-x86_64)
        cd liquidfun/Box2D
        export CC="cl.exe"
        export CXX="cl.exe"
        CXXFLAGS="/Wv:17" $CMAKE -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DBOX2D_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBOX2D_BUILD_EXAMPLES=OFF -DBOX2D_BUILD_UNITTESTS=OFF .
        ninja -j $MAKEJ
        cp -r Box2D $INSTALL_PATH/include
        cp Box2D/*.lib $INSTALL_PATH/lib
        ;;
    macosx-*)
        cd liquidfun/Box2D
        $CMAKE -G "Xcode" -DBOX2D_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBOX2D_BUILD_SHARED=ON -DBOX2D_BUILD_EXAMPLES=OFF -DBOX2D_BUILD_UNITTESTS=OFF -DCMAKE_MACOSX_RPATH=ON .
        xcodebuild -project Box2D.xcodeproj -configuration Release -target install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd $INSTALL_PATH
cd ..
