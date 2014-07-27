if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

LIBDC1394_VERSION=2.2.2
download http://downloads.sourceforge.net/project/libdc1394/libdc1394-2/$LIBDC1394_VERSION/libdc1394-$LIBDC1394_VERSION.tar.gz libdc1394-$LIBDC1394_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
tar -xzvf ../libdc1394-$LIBDC1394_VERSION.tar.gz
cd libdc1394-$LIBDC1394_VERSION

case $PLATFORM in
    linux-x86)
        CC="gcc -m32" ./configure --prefix=$INSTALL_PATH
        make -j4
        make install-strip
        ;;
    linux-x86_64)
        CC="gcc -m64" ./configure --prefix=$INSTALL_PATH
        make -j4
        make install-strip
        ;;
    macosx-*)
        LIBUSB_CFLAGS=-I/usr/local/include/libusb-1.0/ LIBUSB_LIBS=-L/usr/local/lib/ ./configure --prefix=$INSTALL_PATH
        make -j4
        make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
