if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

ARTOOLKITPLUS_VERSION=2.3.0
download https://launchpad.net/artoolkitplus/trunk/$ARTOOLKITPLUS_VERSION/+download/ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2 ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2

tar -xjvf ARToolKitPlus-$ARTOOLKITPLUS_VERSION.tar.bz2
patch -Np1 -d ARToolKitPlus-$ARTOOLKITPLUS_VERSION < ../ARToolKitPlus-$ARTOOLKITPLUS_VERSION.patch
mkdir ARToolKitPlus-$ARTOOLKITPLUS_VERSION/build_$PLATFORM
cd ARToolKitPlus-$ARTOOLKITPLUS_VERSION/build_$PLATFORM

case $PLATFORM in
    android-arm)
        cmake -DCMAKE_TOOLCHAIN_FILE=android-arm.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ANDROID_NDK/../local/" -DLIB_SUFFIX="/armeabi/" ..
        make -j4
        make install
        ;;
     android-x86)
        cmake -DCMAKE_TOOLCHAIN_FILE=android-x86.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ANDROID_NDK/../local/" -DLIB_SUFFIX="/x86/" ..
        make -j4
        make install
        ;;
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" cmake -DCMAKE_BUILD_TYPE=Release -DLIB_SUFFIX=32 ..
        make -j4
        sudo make install
        ;;
    linux-x86_64)
        cmake -DCMAKE_BUILD_TYPE=Release -DLIB_SUFFIX=64 ..
        make -j4
        sudo make install
        ;;
    macosx-x86_64)
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j4
        sudo make install
        ;;
    windows-x86)
        cmake -DCMAKE_BUILD_TYPE=Release ..
        nmake
        nmake install
        ;;
    windows-x86_64)
        cmake -DCMAKE_BUILD_TYPE=Release ..
        nmake
        nmake install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
