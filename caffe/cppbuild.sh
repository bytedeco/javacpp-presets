#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" caffe
    popd
    exit
fi

case $PLATFORM in
    linux-x86)
        export CPU_ONLY=1
        export CC="$OLDCC -m32"
        export CXX="$OLDCXX -m32"
        export FC="$OLDFC -m32"
        export TOOLSET=`echo $OLDCC | sed 's/\([a-zA-Z]*\)\([0-9]\)\([0-9]\)/\1-\2.\3/'`
        export BINARY=32
        export BLAS=open
        ;;
    linux-x86_64)
        export CPU_ONLY=0
        export CC="$OLDCC -m64"
        export CXX="$OLDCXX -m64"
        export FC="$OLDFC -m64"
        export TOOLSET=`echo $OLDCC | sed 's/\([a-zA-Z]*\)\([0-9]\)\([0-9]\)/\1-\2.\3/'`
        export BINARY=64
        export BLAS=open
        ;;
    macosx-*)
        export CPU_ONLY=0
        export CC="clang"
        export CXX="clang++"
        export LDFLAGS="-undefined dynamic_lookup"
        export TOOLSET="clang"
        export BINARY=64
        export BLAS=atlas
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

GLOG=0.3.4
GFLAGS=2.1.2
PROTO=2.6.1
LEVELDB=1.18
SNAPPY=1.1.3
LMDB=0.9.17
BOOST=1_59_0
HDF5=1.8.16
OPENBLAS=0.2.15
CAFFE_VERSION=master

download https://github.com/google/glog/archive/v$GLOG.tar.gz glog-$GLOG.tar.gz
download https://github.com/gflags/gflags/archive/v$GFLAGS.tar.gz gflags-$GFLAGS.tar.gz
download https://github.com/google/protobuf/releases/download/v$PROTO/protobuf-$PROTO.tar.gz protobuf-$PROTO.tar.gz
download https://github.com/google/leveldb/archive/v$LEVELDB.tar.gz leveldb-$LEVELDB.tar.gz
download https://github.com/google/snappy/releases/download/$SNAPPY/snappy-$SNAPPY.tar.gz snappy-$SNAPPY.tar.gz
download https://github.com/LMDB/lmdb/archive/LMDB_$LMDB.tar.gz lmdb-LMDB_$LMDB.tar.gz
download http://downloads.sourceforge.net/project/boost/boost/${BOOST//_/.}/boost_$BOOST.tar.gz boost_$BOOST.tar.gz
download http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-$HDF5/src/hdf5-$HDF5.tar.bz2 hdf5-$HDF5.tar.bz2
download https://github.com/xianyi/OpenBLAS/archive/v$OPENBLAS.tar.gz OpenBLAS-$OPENBLAS.tar.gz
download https://github.com/BVLC/caffe/archive/$CAFFE_VERSION.tar.gz caffe-$CAFFE_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin

OPENCV_PATH="$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/"

echo "Decompressing archives"
tar --totals -xzf ../glog-$GLOG.tar.gz
tar --totals -xzf ../gflags-$GFLAGS.tar.gz
tar --totals -xzf ../protobuf-$PROTO.tar.gz
tar --totals -xzf ../leveldb-$LEVELDB.tar.gz
tar --totals -xzf ../snappy-$SNAPPY.tar.gz
tar --totals -xzf ../lmdb-LMDB_$LMDB.tar.gz
tar --totals -xzf ../boost_$BOOST.tar.gz
tar --totals -xjf ../hdf5-$HDF5.tar.bz2
tar --totals -xzf ../OpenBLAS-$OPENBLAS.tar.gz
tar --totals -xzf ../caffe-$CAFFE_VERSION.tar.gz

export CFLAGS="-fPIC"
export CXXFLAGS="-fPIC"

cd glog-$GLOG
./configure "--prefix=$INSTALL_PATH" --disable-shared
make -j $MAKEJ
make install
cd ..

cd gflags-$GFLAGS
mkdir -p build
cd build
"$CMAKE" -DBUILD_SHARED_LIBS=OFF "-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH" ..
make -j $MAKEJ
make install
cd ../..

cd protobuf-$PROTO
./configure "--prefix=$INSTALL_PATH" --disable-shared
make -j $MAKEJ
make install
cd ..

cd leveldb-$LEVELDB
make -j $MAKEJ
cp -a libleveldb.a "$INSTALL_PATH/lib"
cp -a include/leveldb "$INSTALL_PATH/include/"
cd ..

cd snappy-$SNAPPY
./configure "--prefix=$INSTALL_PATH" --disable-shared
make -j $MAKEJ
make install
cd ..

cd lmdb-LMDB_$LMDB/libraries/liblmdb
make -j $MAKEJ "CC=$CC" "XCFLAGS=$CFLAGS" "CPPFLAGS=$CXXFLAGS"
cp -a lmdb.h ../../../include/
cp -a liblmdb.a ../../../lib/
cd ../../..

cd boost_$BOOST
./bootstrap.sh --with-libraries=filesystem,system,thread
./b2 install "--prefix=$INSTALL_PATH" "address-model=$BINARY" link=static "toolset=$TOOLSET" "cxxflags=$CXXFLAGS"
cd ..
ln -sf libboost_thread.a lib/libboost_thread-mt.a

cd hdf5-$HDF5
LDFLAGS= ./configure "--prefix=$INSTALL_PATH" --disable-shared
make -j $MAKEJ
make install
cd ..

# OSX has Accelerate
if [[ $PLATFORM != macosx-* ]]; then
    # blas (requires fortran, e.g. sudo yum install gcc-gfortran)
    cd OpenBLAS-$OPENBLAS
    make -j $MAKEJ "CC=$CC" "FC=$FC" BINARY=$BINARY NO_SHARED=1
    make install "PREFIX=$INSTALL_PATH" NO_SHARED=1
    cd ..
fi

cd caffe-$CAFFE_VERSION
cp Makefile.config.example Makefile.config
export PATH=../bin:$PATH
export CXXFLAGS="-I../include -I$OPENCV_PATH/include"
export NVCCFLAGS="-I../include -I$OPENCV_PATH/include"
export LINKFLAGS="-L../lib -L$OPENCV_PATH/lib"
make -j $MAKEJ BLAS=$BLAS DISTRIBUTE_DIR=.. lib
# Manual deploy to avoid Caffe's python build
mkdir -p ../include/caffe/proto
cp -a include/caffe/* ../include/caffe/
cp -a build/src/caffe/proto/caffe.pb.h ../include/caffe/proto
cp -a build/lib/libcaffe.so ../lib

cd ../..
