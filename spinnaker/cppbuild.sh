#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" spinnaker
    popd
    exit
fi

SPINNAKER_VERSION=1.27.0.48

case $PLATFORM in
    linux-arm*)
        if [[ ! -f "../../downloads/spinnaker.${SPINNAKER_VERSION}_armhf.tar.gz" ]]; then
            echo "Please place spinnaker.${SPINNAKER_VERSION}_armhf.tar.gz in the downloads directory"
            exit 1
        fi
        echo "Decompressing archives..."
        tar -xzf ../../downloads/spinnaker.${SPINNAKER_VERSION}_armhf.tar.gz
        rm -Rf $PLATFORM
        mv spinnaker.${SPINNAKER_VERSION}_armhf $PLATFORM
        mv $PLATFORM/lib/C/* $PLATFORM/lib
        mv $PLATFORM/include/C/* $PLATFORM/include
        ;;
    linux-x86*)
        if [[ ! -d "/usr/include/spinnaker/" ]]; then
            echo "Please install Spinnaker under the default installation directory"
            exit 1
        fi
        ;;
    windows-*)
        if [[ ! -d "/C/Program Files/Spinnaker/" ]] && [[ ! -d "/C/Program Files (x86)/Spinnaker/" ]] &&
           [[ ! -d "/C/Program Files/Point Grey Research/" ]] && [[ ! -d "/C/Program Files (x86)/Point Grey Research/" ]]; then
            echo "Please install Spinnaker under the default installation directory"
            exit 1
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
