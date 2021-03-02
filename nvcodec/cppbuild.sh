#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" nvcodec
    popd
    exit
fi

case $PLATFORM in
    windows-x86_64)
        if [[ ! -d "/C/Program Files/NVIDIA GPU Computing Toolkit/Video_Codec_SDK_10.0.26/" ]]; then
            echo "Please install Video_Codec_SDK_10.0.26 under the default installation directory"
            exit 1
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..