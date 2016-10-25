#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mxnet
    popd
    exit
fi

case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        export BLAS="openblas"
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        export BLAS="openblas"
        ;;
    macosx-*)
        export CC="clang-omp"
        export CXX="clang-omp++"
        export BLAS="apple"
        ;;
	windows-*)
		return 0
		mkdir -p $PLATFORM
		cd $PLATFORM	
		INSTALL_PATH=`pwd`		
		
		# clone mxnet and its submodules
 		git clone https://github.com/dmlc/mxnet.git --recursive
		cd mxnet
		
		# checkout a stable commit
		git checkout a5aeb0c43028f41863a8148eb3ecf30e90caad1e
		
		# configure the build
		export OPENCV_DIR="$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/" # override the system variable
		USE_X="-DUSE_CUDA=OFF -DUSE_CUDNN=OFF -DUSE_OPENCV=ON"
		OPENCV="-DOpenCV_DIR=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/ -DOpenCV_CONFIG_PATH=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/ -DOpenCV_3RDPARTY_LIB_DIR_DBG=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/lib -DOpenCV_3RDPARTY_LIB_DIR_OPT=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/lib -DOpenCV_LIB_DIR_DBG=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/lib -DOpenCV_LIB_DIR_OPT=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM//lib"
		OPENBLAS="-DOpenBLAS_INCLUDE_DIR=$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/include/ -DOpenBLAS_LIB=$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/lib/libopenblas.dll.a"        
		"$CMAKE" -G "Visual Studio 12 2013 Win64" -DCMAKE_INSTALL_PREFIX=.. $USE_X $OPENBLAS $OPENCV
		
		# build the project
		MSBuild.exe ALL_BUILD.vcxproj //p:Configuration=Release
		
		# copy include files
		mkdir ../include
		cp -a include/mxnet dmlc-core/include/dmlc mshadow/mshadow ../include
		
		# copy binary files
		mkdir ../bin
		cp -a Release/* ../bin
		
		# copy library files
		mkdir ../lib
		cp -a Release/* ../lib
		cd ../..
		return 0
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

MXNET_VERSION=master
download https://github.com/dmlc/dmlc-core/archive/master.tar.gz dmlc-core-$MXNET_VERSION.tar.gz
download https://github.com/dmlc/mshadow/archive/master.tar.gz mshadow-$MXNET_VERSION.tar.gz
download https://github.com/dmlc/ps-lite/archive/master.tar.gz ps-lite-$MXNET_VERSION.tar.gz
download https://github.com/dmlc/mxnet/archive/master.tar.gz mxnet-$MXNET_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
tar -xzvf ../dmlc-core-$MXNET_VERSION.tar.gz
tar -xzvf ../mshadow-$MXNET_VERSION.tar.gz
tar -xzvf ../ps-lite-$MXNET_VERSION.tar.gz
tar -xzvf ../mxnet-$MXNET_VERSION.tar.gz
cd mxnet-$MXNET_VERSION
rmdir dmlc-core mshadow ps-lite || true
ln -snf ../dmlc-core-$MXNET_VERSION dmlc-core
ln -snf ../mshadow-$MXNET_VERSION mshadow
ln -snf ../ps-lite-$MXNET_VERSION ps-lite

export PKG_CONFIG_PATH="$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/lib/pkgconfig/"
export C_INCLUDE_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/include/"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
export LIBRARY_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/lib/"

make -j $MAKEJ CC="$CC" CXX="$CXX" USE_BLAS="$BLAS"
cp -a include lib ../dmlc-core-$MXNET_VERSION/include ..
cp -a ../mshadow-$MXNET_VERSION/mshadow ../include

cd ../..
