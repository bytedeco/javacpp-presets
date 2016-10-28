#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mxnet
    popd
    exit
fi

MXNET_VERSION=a5aeb0c43028f41863a8148eb3ecf30e90caad1e
DMLC_CORE_VERSION=bf321638b22d1d33bb36775e925f7b43b22db688
MSHADOW_VERSION=223b45a5cedf126a50b6c8ca4c82ede8c81874e0
PSLITE_VERSION=36b015ffd51c0f7062bba845f01164c0433dc6b3
download https://github.com/dmlc/dmlc-core/archive/$DMLC_CORE_VERSION.tar.gz dmlc-core-$DMLC_CORE_VERSION.tar.gz
download https://github.com/dmlc/mshadow/archive/$MSHADOW_VERSION.tar.gz mshadow-$MSHADOW_VERSION.tar.gz
download https://github.com/dmlc/ps-lite/archive/$PSLITE_VERSION.tar.gz ps-lite-$PSLITE_VERSION.tar.gz
download https://github.com/dmlc/mxnet/archive/$MXNET_VERSION.tar.gz mxnet-$MXNET_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

tar -xzvf ../dmlc-core-$DMLC_CORE_VERSION.tar.gz
tar -xzvf ../mshadow-$MSHADOW_VERSION.tar.gz
tar -xzvf ../ps-lite-$PSLITE_VERSION.tar.gz
tar -xzvf ../mxnet-$MXNET_VERSION.tar.gz
cd mxnet-$MXNET_VERSION
rmdir dmlc-core mshadow ps-lite || true
ln -snf ../dmlc-core-$DMLC_CORE_VERSION dmlc-core
ln -snf ../mshadow-$MSHADOW_VERSION mshadow
ln -snf ../ps-lite-$PSLITE_VERSION ps-lite

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
	windows-x86_64)
	
        # configure the build
        export OPENCV_DIR="$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/" # override the system variable
        USE_X="-DUSE_CUDA=OFF -DUSE_CUDNN=OFF -DUSE_OPENCV=ON"
        OPENCV="-DOpenCV_DIR=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/ -DOpenCV_CONFIG_PATH=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/ -DOpenCV_3RDPARTY_LIB_DIR_DBG=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/lib -DOpenCV_3RDPARTY_LIB_DIR_OPT=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/lib -DOpenCV_LIB_DIR_DBG=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/lib -DOpenCV_LIB_DIR_OPT=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM//lib"
        OPENBLAS="-DOpenBLAS_INCLUDE_DIR=$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/include/ -DOpenBLAS_LIB=$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/lib/libopenblas.dll.a"        
        "$CMAKE" -G "Visual Studio 12 2013 Win64" $USE_X $OPENBLAS $OPENCV
        
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
		
		# finish
        cd ../..
        return 0
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

export PKG_CONFIG_PATH="$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/lib/pkgconfig/"
export C_INCLUDE_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/include/"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
export LIBRARY_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/lib/"

make -j $MAKEJ CC="$CC" CXX="$CXX" USE_BLAS="$BLAS"
cp -a include lib ../dmlc-core-$DMLC_CORE_VERSION/include ..
cp -a ../mshadow-$MSHADOW_VERSION/mshadow ../include

cd ../..
