#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" caffe
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    echo TODO
else
    GLOG=0.3.4
	GFLAGS=2.1.2
    PROTO=2.6.1
    LEVELDB=1.18
    SNAPPY=1.1.3
    LMDB=0.9.15
    BOOST=1_57_0 # also change the download link, 1.58 incompatible with OSX
    HDF5=1.8.15-patch1
    BLAS=0.2.14
    MAKEJ="${MAKEJ:-4}"

    download https://github.com/google/glog/archive/v$GLOG.tar.gz v$GLOG.tar.gz
    download https://github.com/gflags/gflags/archive/v$GFLAGS.tar.gz v$GFLAGS.tar.gz
    download https://github.com/google/protobuf/releases/download/v$PROTO/protobuf-$PROTO.tar.gz protobuf-$PROTO.tar.gz
    download https://github.com/google/leveldb/archive/v$LEVELDB.tar.gz v$LEVELDB.tar.gz
    download https://github.com/google/snappy/releases/download/$SNAPPY/snappy-$SNAPPY.tar.gz snappy-$SNAPPY.tar.gz
    download https://github.com/LMDB/lmdb/archive/LMDB_$LMDB.tar.gz LMDB_$LMDB.tar.gz
    download http://iweb.dl.sourceforge.net/project/boost/boost/1.57.0/boost_$BOOST.tar.gz boost_$BOOST.tar.gz
    download http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-$HDF5.tar.bz2 hdf5-$HDF5.tar.bz2
    download https://github.com/xianyi/OpenBLAS/archive/v$BLAS.tar.gz v$BLAS.tar.gz

    mkdir -p $PLATFORM
    cd $PLATFORM
    INSTALL_PATH=`pwd`
    mkdir -p include
    mkdir -p lib

    echo "Decompressing archives"
    tar --totals -xzf ../v$GLOG.tar.gz
    tar --totals -xzf ../v$GFLAGS.tar.gz
    tar --totals -xzf ../protobuf-$PROTO.tar.gz
    tar --totals -xzf ../v$LEVELDB.tar.gz
    tar --totals -xzf ../snappy-$SNAPPY.tar.gz
    tar --totals -xzf ../LMDB_$LMDB.tar.gz
    tar --totals -xzf ../boost_$BOOST.tar.gz
    tar --totals -xjf ../hdf5-$HDF5.tar.bz2
    tar --totals -xzf ../v$BLAS.tar.gz
fi

cd glog-$GLOG
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ
make install
cd ..

cd gflags-$GFLAGS
mkdir -p build
cd build
export CXXFLAGS="-fPIC" && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
make -j $MAKEJ
make install
cd ../..

cd protobuf-$PROTO
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ
make install
cd ..

cd leveldb-$LEVELDB
make -j $MAKEJ
cp -a libleveldb.* $INSTALL_PATH/lib
cp -a include/leveldb $INSTALL_PATH/include/
cd ..

cd snappy-$SNAPPY
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ
make install
cd ..

cd lmdb-LMDB_$LMDB/libraries/liblmdb
make -j $MAKEJ
cp -a lmdb.h ../../../include/
cp -a liblmdb.so ../../../lib/
cd ../../..

cd boost_$BOOST
./bootstrap.sh --with-libraries=system,thread
./b2 install --prefix=$INSTALL_PATH
cd ..

cd hdf5-$HDF5
./configure --prefix=$INSTALL_PATH
make -j $MAKEJ
make install
cd ..

# OSX has Accelerate
if [[ $PLATFORM != macosx-* ]]; then
    # blas (requires fortran, e.g. sudo yum install gcc-gfortran)
	cd OpenBLAS-$BLAS
    # CentOS compiler version can't compile AVX2 instructions, TODO update compiler
	make -j $MAKEJ NO_AVX2=1
	make install PREFIX=$INSTALL_PATH
	cd ..
fi

CAFFE_VERSION=master
download https://github.com/BVLC/caffe/archive/master.zip ../caffe-$CAFFE_VERSION.zip
unzip ../caffe-$CAFFE_VERSION.zip
cd caffe-$CAFFE_VERSION

cp Makefile.config.example Makefile.config
printf "\n" >> Makefile.config
printf "INCLUDE_DIRS += ../include ../../../../opencv/cppbuild/linux-x86_64/include\n" >> Makefile.config
printf "LIBRARY_DIRS += ../lib ../../../../opencv/cppbuild/linux-x86_64/lib\n" >> Makefile.config
printf "BLAS := open\n" >> Makefile.config
export PATH=../bin:$PATH
export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH
CC="gcc -m64" CXX="g++ -m64" BLAS=open DISTRIBUTE_DIR=.. make -j $MAKEJ lib
# Manual deploy to avoid Caffe's python build
mkdir -p ../include/caffe/proto
cp -a include/caffe/* ../include/caffe/
cp -a build/src/caffe/proto/caffe.pb.h ../include/caffe/proto
cp -a build/lib/libcaffe.so ../lib

cd ../..
