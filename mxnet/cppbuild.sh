#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mxnet
    popd
    exit
fi

export ADD_CFLAGS="-DMXNET_USE_LAPACK=1"
export ADD_LDFLAGS=
export USE_OPENMP=1
export CUDA_ARCH=-arch=sm_35
export USE_CUDA=0
export USE_CUDNN=0
export USE_NCCL=0
export USE_CUDA_PATH=
export NVCC=
export USE_MKLDNN=1
if [[ "$EXTENSION" == *gpu ]]; then
    export ADD_CFLAGS="$ADD_CFLAGS -DMXNET_USE_CUDA=1"
    export USE_CUDA=1
    export USE_CUDNN=1
    if [[ "$PLATFORM" == linux* ]]; then
        export USE_NCCL=1
    fi
    export USE_CUDA_PATH="/usr/local/cuda"
    export NVCC="$USE_CUDA_PATH/bin/nvcc"
    export ACTUAL_GCC_HOST_COMPILER_PATH=$(which -a gcc | grep -v /ccache/ | head -1) # skip ccache
fi

MXNET_VERSION=1.9.0
download https://downloads.apache.org/incubator/mxnet/$MXNET_VERSION/apache-mxnet-src-$MXNET_VERSION-incubating.tar.gz apache-mxnet-src-$MXNET_VERSION-incubating.tar.gz
#download https://github.com/apache/incubator-mxnet/releases/download/$MXNET_VERSION/apache-mxnet-src-$MXNET_VERSION-incubating.tar.gz apache-mxnet-src-$MXNET_VERSION-incubating.tar.gz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`

if [[ -f "lib/libmxnet.so" ]]; then
    echo libmxnet.so already exists!
    return 0
elif [[ -f "bin/libmxnet.dll" ]]; then
    echo libmxnet.dll already exists!
    return 0
fi

OPENCV_PATH="$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/"
OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -d "$P/include/opencv2" ]]; then
            OPENCV_PATH="$P"
        elif [[ -f "$P/include/openblas_config.h" ]]; then
            OPENBLAS_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

echo "Decompressing archives... (ignore any errors)"
tar --totals -xzf ../apache-mxnet-src-$MXNET_VERSION-incubating.tar.gz || true

cd apache-mxnet-src-$MXNET_VERSION-incubating

# https://github.com/apache/incubator-mxnet/pull/20207
patch -Np1 < ../../../mxnet.patch

# upgrade MKL-DNN
#sedinplace 's/0.18.1/0.19/g' 3rdparty/mkldnn/CMakeLists.txt
#sedinplace 's/0.18/0.19/g' 3rdparty/mkldnn/scripts/prepare_mkl.bat 3rdparty/mkldnn/scripts/prepare_mkl.sh
#sedinplace 's/2019.0.3.20190220/2019.0.5.20190502/g' 3rdparty/mkldnn/scripts/prepare_mkl.bat 3rdparty/mkldnn/scripts/prepare_mkl.sh

# actually use LAPACK from OpenBLAS
sedinplace 's/USE_LAPACK = 0/USE_LAPACK = 1/g' Makefile
sedinplace '/LDFLAGS += -llapack/d' Makefile

# patch up compile errors
sedinplace "s/cmake/$CMAKE/g" mkldnn.mk
sedinplace 's/-Werror//g' 3rdparty/mkldnn/cmake/platform.cmake
sedinplace '/include_directories("3rdparty\/nvidia_cub")/d' CMakeLists.txt
sedinplace 's/kCPU/Context::kCPU/g' src/operator/tensor/elemwise_binary_scalar_op_basic.cc
sedinplace 's:../../src/operator/tensor/:./:g' src/operator/tensor/cast_storage-inl.h
sedinplace 's/-Xcompiler "$(CFLAGS)/"-Xcompiler=$(CFLAGS)/g' Makefile
sedinplace '/CHECK(mshadow::DataType<DType>::kFlag == type_flag_)/{N;N;N;d;}' include/mxnet/tensor_blob.h
sedinplace 's/std::pow/powf/g' src/operator/contrib/multi_lamb.cu
sedinplace 's/round(/roundf(/g' src/operator/*.cu src/operator/contrib/*.cu
sedinplace 's/floor(/floorf(/g' src/operator/*.cu src/operator/contrib/*.cu
sedinplace 's/ceil(/ceilf(/g' src/operator/*.cu src/operator/contrib/*.cu

sedinplace '/#include <opencv2\/opencv.hpp>/a\
#include <opencv2/imgproc/types_c.h>\
' src/io/image_augmenter.h src/io/image_io.cc tools/im2rec.cc
sedinplace 's/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g' tools/im2rec.cc
sedinplace 's/CV_IMWRITE_PNG_COMPRESSION/cv::IMWRITE_PNG_COMPRESSION/g' tools/im2rec.cc
sedinplace 's/CV_IMWRITE_JPEG_QUALITY/cv::IMWRITE_JPEG_QUALITY/g' tools/im2rec.cc

# note: MXNet needs full path to ccache wrappers to use them
case $PLATFORM in
    linux-x86)
        export CC="$(which gcc) -m32"
        export CXX="$(which g++) -m32"
        export BLAS="openblas"
        export USE_INTGEMM=0
        export USE_MKLDNN=0
        ;;
    linux-x86_64)
        export CC="$(which gcc) -m64"
        export CXX="$(which g++) -m64"
        export BLAS="openblas"
        sedinplace "s/-std=c++11/-std=c++14/g" Makefile
        # libmklml_intel.so does not have a SONAME, so libmkldnn.so.0 needs an RPATH to be able to load
#        sedinplace "s/$CMAKE/$CMAKE -DCMAKE_CXX_FLAGS='-Wl,-rpath,\$\$ORIGIN\/'/g" mkldnn.mk
        if [[ -f /usr/local/cuda/bin/nvcccache ]]; then
            export NVCC=/usr/local/cuda/bin/nvcccache
            sedinplace 's/-ccbin $(CXX)/-ccbin $(ACTUAL_GCC_HOST_COMPILER_PATH) -Xcompiler -fPIC -Xcompiler -O3/g' Makefile
        fi
        ;;
    macosx-*)
        # remove harmful changes to rpath
        sedinplace '/install_name_tool/d' Makefile
        export CC="$(which clang)"
        export CXX="$(which clang++)"
        export BLAS="openblas"
        export USE_OPENMP=0
#        export USE_OPENMP=1
#        export ADD_CFLAGS="$ADD_CFLAGS -I/usr/local/include"
#        export ADD_CFLAGS="$ADD_CFLAGS -L/usr/local/lib -lomp"
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"

        # copy include files
        mkdir -p ../include
        cp -RLf include/mxnet 3rdparty/dmlc-core/include/dmlc 3rdparty/mshadow/mshadow ../include

        # configure the build
        mkdir -p ../build
        cd ../build
        USE_X="-DMXNET_CUDA_ARCH=3.5+PTX -DCUDA_ARCH_LIST=3.5+PTX -DUSE_CUDA=$USE_CUDA -DUSE_CUDNN=$USE_CUDNN -DUSE_OPENCV=ON -DUSE_MKLDNN=$USE_MKLDNN"
        OPENCV="-DOpenCV_DIR=$OPENCV_PATH/ -DOpenCV_CONFIG_PATH=$OPENCV_PATH/"
        OPENBLAS="-DOpenBLAS_INCLUDE_DIR=$OPENBLAS_PATH/include/ -DOpenBLAS_LIB=$OPENBLAS_PATH/lib/openblas.lib"
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release $USE_X $OPENCV $OPENBLAS ../apache-mxnet-src-$MXNET_VERSION-incubating

        # try to rebuild the project without compiler parallelism to avoid "out of heap space"
        ninja -j $MAKEJ || ninja -j 1

        # copy binary files
        mkdir -p ../bin
        cp -f *.dll ../bin

        # copy library files
        mkdir -p ../lib
        cp -f libmxnet.lib ../lib/mxnet.lib
        cp -f mxnet_??.lib ../lib/mxnet.lib || true

        # finish
        cd ../apache-mxnet-src-$MXNET_VERSION-incubating
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

if [[ ! "$PLATFORM" == windows* ]]; then
    export C_INCLUDE_PATH="$OPENBLAS_PATH/include/:$OPENCV_PATH/include/"
    export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
    export LIBRARY_PATH="$OPENBLAS_PATH/:$OPENBLAS_PATH/lib/:$OPENCV_PATH/:$OPENCV_PATH/lib/"

    sed -i="" 's/$(shell pkg-config --cflags opencv)//' Makefile
    sed -i="" 's/$(shell pkg-config --libs opencv)/-lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core/' Makefile
    sed -i="" 's/$(shell pkg-config --cflags $(OPENCV_LIB))//' Makefile
    sed -i="" 's/$(shell pkg-config --libs-only-L $(OPENCV_LIB))//' Makefile
    sed -i="" 's/$(filter -lopencv_imgcodecs -lopencv_highgui, $(shell pkg-config --libs-only-l $(OPENCV_LIB)))/-lopencv_highgui -lopencv_imgcodecs/' Makefile
    make -s -j $MAKEJ CC="$CC" CXX="$CXX" USE_BLAS="$BLAS" USE_OPENMP="$USE_OPENMP" CUDA_ARCH="$CUDA_ARCH" USE_CUDA="$USE_CUDA" USE_CUDNN="$USE_CUDNN" USE_NCCL="$USE_NCCL" USE_CUDA_PATH="$USE_CUDA_PATH" NVCC="$NVCC" USE_MKLDNN="$USE_MKLDNN" USE_F16C=0 ADD_CFLAGS="$ADD_CFLAGS" ADD_LDFLAGS="$ADD_LDFLAGS" lib/libmxnet.a lib/libmxnet.so
    cp -RLf include lib 3rdparty/dmlc-core/include ..
    cp -RLf 3rdparty/mshadow/mshadow ../include
    unset CC
    unset CXX
fi

# add Scala files we cannot get with CMake for Windows
# cd scala-package
# mvn clean install -Dmaven.test.skip
# rm -Rf `find ./ -iname target`
patch -Np1 < ../../../mxnet-scala.patch || true
# use mxnet-scala-cuda.patch to get a couple of CUDA ops,
# but an actual GPU becomes necessary for the build

# copy official JNI functions and adjust include directives
cp -RLf 3rdparty/dlpack/include/dlpack 3rdparty/tvm/nnvm/include/* ../include
cp -f src/common/cuda_utils.h scala-package/init-native/src/main/native/* scala-package/native/src/main/native/* ../include
sedinplace 's:../src/common/::g' ../include/*.cc

# copy/adjust Scala source files and work around loader issue in Base.scala
mkdir -p ../scala/init ../scala/core
cp -RLf scala-package/init/src/main/scala/* ../scala/init
cp -RLf scala-package/macros/src/main/scala/* ../scala/init
cp -RLf scala-package/core/src/main/scala/* ../scala/core
cp -RLf scala-package/infer/src/main/scala/* ../scala/core
cp -RLf scala-package/spark/src/main/scala/* ../scala/core
sedinplace 's/  tryLoadInitLibrary()/  org.bytedeco.javacpp.Loader.load(classOf[org.bytedeco.mxnet.presets.mxnet])/g' ../scala/init/org/apache/mxnet/init/Base.scala
sedinplace 's/  tryLoadLibraryOS("mxnet-scala")/  org.bytedeco.javacpp.Loader.load(classOf[org.bytedeco.mxnet.presets.mxnet])/g' ../scala/core/org/apache/mxnet/Base.scala

# fix library with correct rpath on Mac
case $PLATFORM in
    macosx-*)
        install_name_tool -add_rpath @loader_path/. -id @rpath/libmxnet.so ../lib/libmxnet.so
        ;;
esac

cd ../..
