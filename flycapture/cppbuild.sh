#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" flycapture
    popd
    exit
fi

FLYCAPTURE_VERSION=2.11.3.121

case $PLATFORM in
    linux-arm*)
        if [[ ! -f "../../downloads/flycapture.${FLYCAPTURE_VERSION}_armhf.tar.gz" ]]; then
            echo "Please place flycapture.${FLYCAPTURE_VERSION}_armhf.tar.gz in the downloads directory"
            exit 1
        fi
        echo "Decompressing archives..."
        tar -xzf ../../downloads/flycapture.${FLYCAPTURE_VERSION}_armhf.tar.gz
        rm -Rf $PLATFORM
        mv flycapture.${FLYCAPTURE_VERSION}_armhf $PLATFORM
        mv $PLATFORM/lib/C/* $PLATFORM/lib
        mv $PLATFORM/include/C/* $PLATFORM/include
        ;;
    linux-x86*)
        if [[ ! -d "/usr/include/flycapture/" ]]; then
            echo "Please install FlyCapture under the default installation directory"
            exit 1
        fi
        ;;
    windows-*)
        if [[ ! -d "/C/Program Files/Point Grey Research/" ]]; then
            echo "Please install FlyCapture under the default installation directory"
            exit 1
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
