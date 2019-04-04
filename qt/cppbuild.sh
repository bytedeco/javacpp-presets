#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" qt
    popd
    exit
fi

QT_VERSION=5.12.2
download https://download.qt.io/archive/qt/${QT_VERSION%.*}/$QT_VERSION/single/qt-everywhere-src-$QT_VERSION.tar.xz qt-$QT_VERSION.tar.xz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xf ../qt-$QT_VERSION.tar.xz

cd qt-everywhere-src-$QT_VERSION

# remove stuff not actually available on Mac
sedinplace '/kTLSProtocol13/d' qtbase/src/network/ssl/qsslsocket_mac.cpp

QT_OPTIONS="-prefix .. -qt-zlib -qt-libjpeg -qt-libpng -qt-xcb -qt-freetype -qt-pcre -qt-harfbuzz -opensource -confirm-license -nomake examples -nomake tests -nomake tools -skip qt3d -skip qtactiveqt -skip qtandroidextras -skip qtcanvas3d -skip qtcharts -skip qtconnectivity -skip qtdatavis3d -skip qtdeclarative -skip qtdoc -skip qtgamepad -skip qtgraphicaleffects -skip qtimageformats -skip qtlocation -skip qtmacextras -skip qtmultimedia -skip qtnetworkauth -skip qtpurchasing -skip qtquickcontrols -skip qtquickcontrols2 -skip qtremoteobjects -skip qtscript -skip qtscxml -skip qtsensors -skip qtserialbus -skip qtserialport -skip qtspeech -skip qtsvg -skip qttools -skip qttranslations -skip qtvirtualkeyboard -skip qtwayland -skip qtwebchannel -skip qtwebengine -skip qtwebglplugin -skip websockets -skip qtwebview -skip qtwinextras -skip qtx11extras -skip qtxmlpatterns -no-icu -no-framework -release -silent"

case $PLATFORM in
    linux-x86_64)
        ./configure $QT_OPTIONS
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        ./configure $QT_OPTIONS
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

for f in `find ../plugins`; do cp -a $f ../lib; done

cd ../..
