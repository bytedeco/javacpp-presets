#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mkl
    popd
    exit
fi

case $PLATFORM in
    linux-x86*)
        if [[ ! -d "/opt/intel/mkl/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        ;;
    macosx-*)
        if [[ ! -d "/opt/intel/mkl/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        ;;
    windows-*)
        if [[ ! -d "/C/Program Files (x86)/IntelSWTools/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
