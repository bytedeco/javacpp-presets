if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

FLANDMARK_VERSION=master
download https://github.com/uricamic/flandmark/archive/$FLANDMARK_VERSION.zip flandmark-$FLANDMARK_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
mkdir -p include lib bin
unzip -o ../flandmark-$FLANDMARK_VERSION.zip
cd flandmark-$FLANDMARK_VERSION

OPENCV_PATH=$INSTALL_PATH/../../../opencv/cppbuild/$PLATFORM/

case $PLATFORM in
    android-arm)
        cmake -DCMAKE_TOOLCHAIN_FILE=$INSTALL_PATH/../../android-arm.cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/ -DANDROID_NDK_ABI_NAME=armeabi_v7a
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    android-x86)
        cmake -DCMAKE_TOOLCHAIN_FILE=$INSTALL_PATH/../../android-x86.cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/sdk/native/jni/ -DANDROID_NDK_ABI_NAME=x86
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    macosx-*)
        CXX="g++ -fpermissive" cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/share/OpenCV/
        make -j4 flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.a ../lib
        ;;
    windows-x86)
        cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/opencv/build/
        nmake flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.lib ../lib
        ;;
    windows-x86_64)
        cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_PATH/opencv/build/
        nmake flandmark_static
        cp libflandmark/*.h ../include
        cp libflandmark/*.lib ../lib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
