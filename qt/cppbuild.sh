#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" qt
    popd
    exit
fi

QT_VERSION=5.15.2
download https://download.qt.io/archive/qt/${QT_VERSION%.*}/$QT_VERSION/single/qt-everywhere-src-$QT_VERSION.tar.xz qt-$QT_VERSION.tar.xz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xf ../qt-$QT_VERSION.tar.xz

cd qt-everywhere-src-$QT_VERSION

# remove stuff not actually available on most Macs
sedinplace '/kTLSProtocol13/d' qtbase/src/network/ssl/qsslsocket_mac.cpp
sedinplace '/backtrace_from_fp/d' qtbase/src/corelib/kernel/qcore_mac.mm
sedinplace 's/(__builtin_available(.*)/(0)/g' qtbase/src/corelib/kernel/qcore_mac.mm
sedinplace 's/kIOSurfaceSuccess/KERN_SUCCESS/g' qtbase/src/plugins/platforms/cocoa/qiosurfacegraphicsbuffer.mm
sedinplace 's/QT_MACOS_PLATFORM_SDK_EQUAL_OR_ABOVE(__MAC_10_14)/0/g' qtbase/src/corelib/kernel/qcore_mac.mm qtbase/src/plugins/styles/mac/qmacstyle_mac.mm qtbase/src/plugins/platforms/cocoa/qcocoatheme.mm

QT_OPTIONS="-qt-zlib -qt-libjpeg -qt-libpng -qt-pcre -qt-harfbuzz -opensource -confirm-license -nomake examples -nomake tests -nomake tools -skip qt3d -skip qtactiveqt -skip qtandroidextras -skip qtcanvas3d -skip qtcharts -skip qtconnectivity -skip qtdatavis3d -skip qtdeclarative -skip qtdoc -skip qtgamepad -skip qtgraphicaleffects -skip qtimageformats -skip qtlocation -skip qtmacextras -skip qtmultimedia -skip qtnetworkauth -skip qtpurchasing -skip qtquickcontrols -skip qtquickcontrols2 -skip qtremoteobjects -skip qtscript -skip qtscxml -skip qtsensors -skip qtserialbus -skip qtserialport -skip qtspeech -skip qtsvg -skip qttools -skip qttranslations -skip qtvirtualkeyboard -skip qtwayland -skip qtwebchannel -skip qtwebengine -skip qtwebglplugin -skip websockets -skip qtwebview -skip qtwinextras -skip qtx11extras -skip qtxmlpatterns -no-icu -no-framework -release -silent"

case $PLATFORM in
    linux-x86_64)
        ./configure -prefix .. $QT_OPTIONS -xcb
        make -j $MAKEJ
        make install
        ;;
    macosx-*)
        ./configure -prefix .. $QT_OPTIONS
        make -j $MAKEJ
        make install
        ;;
    windows-x86_64)
        # Qt can only be built from very short paths on Windows
        PLATFORM= cmd.exe //c "mklink /j \\qt ."
        PLATFORM= cmd.exe //c "cd \\qt & configure.bat -prefix $(cygpath -w $INSTALL_PATH) $QT_OPTIONS & nmake & nmake install"
        PLATFORM= cmd.exe //c "rmdir \\qt"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

for f in `find ../plugins`; do cp -a $f ../lib; done

cd ../..
