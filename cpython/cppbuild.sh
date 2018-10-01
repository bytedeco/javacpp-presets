#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" cpython
    popd
    exit
fi

case $PLATFORM in
    linux-*)
        if [[ ! -d "/usr/include/python3.6m/" ]]; then
            echo "Please install Python 3.6 under the default installation directory"
            exit 1
        fi
        ;;
    macosx-*)
        if [[ ! -d "/Library/Frameworks/Python.framework/Versions/3.6/" ]]; then
            echo "Please install Python 3.6 under the default installation directory"
            exit 1
        fi
        ;;
    windows-*)
        if [[ ! -d "/C/Program Files/Python36/" ]]; then
            echo "Please install Python 3.6 under the default installation directory"
            exit 1
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
