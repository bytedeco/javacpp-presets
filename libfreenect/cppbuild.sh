if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

LIBFREENECT_VERSION=0.4.0
download https://github.com/OpenKinect/libfreenect/archive/v$LIBFREENECT_VERSION.zip libfreenect-$LIBFREENECT_VERSION.zip

unzip -o libfreenect-$LIBFREENECT_VERSION.zip

if [[ $PLATFORM == windows* ]]; then
    download http://downloads.sourceforge.net/project/libusb-win32/libusb-win32-releases/1.2.6.0/libusb-win32-bin-1.2.6.0.zip libusb-win32-bin-1.2.6.0.zip
    download ftp://sourceware.org/pub/pthreads-win32/pthreads-w32-2-9-1-release.zip pthreads-w32-2-9-1-release.zip

    unzip -o libusb-win32-bin-1.2.6.0.zip -d C:/
    unzip -o pthreads-w32-2-9-1-release.zip -d C:/pthreads-w32-2-9-1-release/
    patch -Np1 -d libfreenect-$LIBFREENECT_VERSION < ../libfreenect-$LIBFREENECT_VERSION-windows.patch
fi

mkdir libfreenect-$LIBFREENECT_VERSION/build_$PLATFORM
cd libfreenect-$LIBFREENECT_VERSION/build_$PLATFORM

case $PLATFORM in
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" cmake -DCMAKE_BUILD_TYPE=Release -DLIB_SUFFIX=32 -DBUILD_AUDIO=ON ..
        make -j4
        sudo make install
        ;;
    linux-x86_64)
        cmake -DCMAKE_BUILD_TYPE=Release -DLIB_SUFFIX=64 -DBUILD_AUDIO=ON ..
        make -j4
        sudo make install
        ;;
    macosx-x86_64)
        cmake -DCMAKE_BUILD_TYPE=Release -DLIB_SUFFIX= -DBUILD_AUDIO=ON -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF ..
        make -j4
        sudo make install
        ;;
    windows-x86)
        cmake -DCMAKE_BUILD_TYPE=Release -DLIBUSB_1_INCLUDE_DIR="C:/libusb-win32-bin-1.2.6.0/include" -DLIBUSB_1_LIBRARY="C:/libusb-win32-bin-1.2.6.0/lib/msvc/libusb.lib" -DTHREADS_PTHREADS_INCLUDE_DIR="C:/pthreads-w32-2-9-1-release/Pre-built.2/include" -DTHREADS_PTHREADS_WIN32_LIBRARY="C:/pthreads-w32-2-9-1-release/Pre-built.2/lib/x86/pthreadVC2.lib" -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF ..
        nmake
        nmake install
        ;;
    windows-x86_64)
        cmake -DCMAKE_BUILD_TYPE=Release -DLIBUSB_1_INCLUDE_DIR="C:/libusb-win32-bin-1.2.6.0/include" -DLIBUSB_1_LIBRARY="C:/libusb-win32-bin-1.2.6.0/lib/msvc_x64/libusb.lib" -DTHREADS_PTHREADS_INCLUDE_DIR="C:/pthreads-w32-2-9-1-release/Pre-built.2/include" -DTHREADS_PTHREADS_WIN32_LIBRARY="C:/pthreads-w32-2-9-1-release/Pre-built.2/lib/x64/pthreadVC2.lib" -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF ..
        nmake
        nmake install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
