#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" cuda
    popd
    exit
fi

case $PLATFORM in
    linux-*)
        if [[ ! -d "/usr/local/cuda/" ]]; then
            echo "Please install CUDA under the default installation directory"
            exit 1
        fi
        ;;
    macosx-*)
        if [[ ! -d "/usr/local/cuda/" ]]; then
            echo "Please install CUDA under the default installation directory"
            exit 1
        fi
        ;;
    windows-*)
        if [[ ! -d "/C/Program Files/NVIDIA GPU Computing Toolkit/CUDA/" ]]; then
            echo "Please install CUDA under the default installation directory"
            exit 1
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
