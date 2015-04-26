#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" flycapture
    popd
    exit
fi

case $PLATFORM in
    linux-*)
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
