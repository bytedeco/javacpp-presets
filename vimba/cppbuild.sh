#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" vimba
    popd
    exit
fi

VIMBA_VERSION=2.1.3
# TODO: make the 'Linux' below dynamic
download https://www.alliedvision.com/fileadmin/content/software/software/Vimba/Vimba_v${VIMBA_VERSION}_Linux.tgz  vimba-$VIMBA_VERSION.tar.gz

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"

INSTALL_PATH=`pwd`
INCLUDE_PATH="$INSTALL_PATH/include"
LIB_PATH="$INSTALL_PATH/lib"

echo "Decompressing archives..."
tar --totals -xzf ../vimba-$VIMBA_VERSION.tar.gz

case $PLATFORM in
#    linux-arm*)
#        if [[ ! -f "../../downloads/flycapture.${FLYCAPTURE_VERSION}_armhf.tar.gz" ]]; then
#            echo "Please place flycapture.${FLYCAPTURE_VERSION}_armhf.tar.gz in the downloads directory"
#            exit 1
#        fi
#        echo "Decompressing archives..."
#        tar -xzf ../../downloads/flycapture.${FLYCAPTURE_VERSION}_armhf.tar.gz
#        rm -Rf $PLATFORM
#        mv flycapture.${FLYCAPTURE_VERSION}_armhf $PLATFORM
#        mv $PLATFORM/lib/C/* $PLATFORM/lib
#        mv $PLATFORM/include/C/* $PLATFORM/include
#        ;;
    linux-x86*)
        cd Vimba_*/VimbaCPP/Build/Make
        make -j $MAKEJ

        rm -rf $INCLUDE_PATH

        mkdir -p "$INCLUDE_PATH/VimbaC"
        cp -r ../../../VimbaC/Include "$INCLUDE_PATH/VimbaC/"

        mkdir -p "$INCLUDE_PATH/VimbaCPP"
        cp -r ../../../VimbaCPP/Include "$INCLUDE_PATH/VimbaCPP/"

        rm -rf $LIB_PATH
        mkdir -p $LIB_PATH
        cp -r dynamic/x86_64bit/* "$LIB_PATH/"

        ;;
#    windows-*)
#        if [[ ! -d "/C/Program Files/Point Grey Research/" ]]; then
#            echo "Please install FlyCapture under the default installation directory"
#            exit 1
#        fi
#        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
