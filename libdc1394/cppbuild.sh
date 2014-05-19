if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

LIBDC1394_VERSION=2.2.2
download http://downloads.sourceforge.net/project/libdc1394/libdc1394-2/$LIBDC1394_VERSION/libdc1394-$LIBDC1394_VERSION.tar.gz libdc1394-$LIBDC1394_VERSION.tar.gz

tar -xzvf libdc1394-$LIBDC1394_VERSION.tar.gz
mv libdc1394-$LIBDC1394_VERSION libdc1394-$LIBDC1394_VERSION-$PLATFORM
cd libdc1394-$LIBDC1394_VERSION-$PLATFORM

case $PLATFORM in
    linux-x86)
        CC="gcc -m32" ./configure --libdir=/usr/local/lib32/
        make -j4
        sudo make install-strip
        ;;
    linux-x86_64)
        ./configure --libdir=/usr/local/lib64/
        make -j4
        sudo make install-strip
        ;;
    macosx-x86_64)
        LIBUSB_CFLAGS=-I/usr/local/include/libusb-1.0/ LIBUSB_LIBS=-I/usr/local/lib/ ./configure
        make -j4
        sudo make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ..
