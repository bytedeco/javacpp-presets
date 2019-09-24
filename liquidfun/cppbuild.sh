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
        CXXFLAGS="/Wv:17" $CMAKE -G "Visual Studio 15 2017" -DBOX2D_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBOX2D_BUILD_SHARED=ON -DBOX2D_BUILD_EXAMPLES=OFF -DBOX2D_BUILD_UNITTESTS=OFF .
        MSBuild.exe Box2D.sln //p:Configuration=Release //p:Platform=Win32 //p:CL_MPCount=$MAKEJ
        cp -r Box2D $INSTALL_PATH/include
        cp Box2D/Release/*.lib $INSTALL_PATH/lib
        cp Box2D/Release/*.dll $INSTALL_PATH/bin
        ;;
    windows-x86_64)
        cd liquidfun/Box2D
        CXXFLAGS="/Wv:17" $CMAKE -G "Visual Studio 15 2017 Win64" -DBOX2D_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBOX2D_BUILD_SHARED=ON -DBOX2D_BUILD_EXAMPLES=OFF -DBOX2D_BUILD_UNITTESTS=OFF .
        MSBuild.exe Box2D.sln //p:Configuration=Release //p:Platform=x64 //p:CL_MPCount=$MAKEJ
        cp -r Box2D $INSTALL_PATH/include
        cp Box2D/Release/*.lib $INSTALL_PATH/lib
        cp Box2D/Release/*.dll $INSTALL_PATH/bin
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
