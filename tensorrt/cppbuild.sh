#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tensorrt
    popd
    exit
fi

case $PLATFORM in
    linux-x86_64)
        if [[ ! -f "/usr/include/x86_64-linux-gnu/NvInfer.h" ]] && [[ ! -d "/usr/local/tensorrt/" ]]; then
            echo "Please install TensorRT under the default installation directory or in /usr/local/tensorrt/"
            exit 1
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
