#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" nvcodec
    popd
    exit
fi

case $PLATFORM in
    linux-arm64)
        if [[ ! -d "/usr/local/videocodecsdk/" ]]; then
            echo "Please install the Video Codec SDK in /usr/local/videocodecsdk/"
            exit 1
        fi
        ;;
    linux-ppc64le)
        if [[ ! -d "/usr/local/videocodecsdk/" ]]; then
            echo "Please install the Video Codec SDK in /usr/local/videocodecsdk/"
            exit 1
        fi
        ;;
    linux-x86_64)
        if [[ ! -d "/usr/local/videocodecsdk/" ]]; then
            echo "Please install the Video Codec SDK in /usr/local/videocodecsdk/"
            exit 1
        fi
        ;;
    windows-x86_64)
        if [[ ! -d "C:/Program Files/NVIDIA GPU Computing Toolkit/VideoCodecSDK/" ]]; then
            echo "Please install the Video Codec SDK in C:/Program Files/NVIDIA GPU Computing Toolkit/VideoCodecSDK/"
            exit 1
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
