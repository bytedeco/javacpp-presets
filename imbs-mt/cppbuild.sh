#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" imbs-mt
    popd
    exit
fi

IMBS_VERSION=master
download https://github.com/dbloisi/imbs-mt/archive/master.zip imbs-mt-$IMBS_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
unzip -o ../imbs-mt-$IMBS_VERSION.zip
cd imbs-mt-$IMBS_VERSION
patch CMakeLists.txt < ../../../imbs-mt.patch || true

OPENCV_PATH=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -d "$P/include/opencv2" ]]; then
            OPENCV_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

case $PLATFORM in
    linux-x86_64)
        CXX="g++ -m64 -fPIC" $CMAKE -j $MAKEJ -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/lib/cmake/opencv4/ .
        make -j $MAKEJ
        cp *.h ../include
        cp *.hpp ../include
        cp *.a ../lib
        ;;
    windows-x86_64)
	"$CMAKE" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DOpenCV_DIR=$INSTALL_PATH/../../../opencv/target/native/org/bytedeco/opencv/windows-x86_64/x64/vc14/lib/ .
        nmake
	cp *.lib ../lib
        cp *.h ../include
	cp *.hpp ../include	
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
