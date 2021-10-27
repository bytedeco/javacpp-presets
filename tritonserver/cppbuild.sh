#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tritonserver
    popd
    exit
fi

case $PLATFORM in
    linux-arm64)
        if [[ ! -f "/opt/tritonserver/include/triton/core/tritonserver.h" ]] && [[ ! -d "/opt/tritonserver/lib/" ]]; then
            echo "Please make sure library and include files exist"
            exit 1
        fi
        ;;
    linux-x86_64)
        if [[ ! -f "/opt/tritonserver/include/triton/core/tritonserver.h" ]] && [[ ! -d "/opt/tritonserver/lib/" ]]; then
            echo "Please make sure library and include files exist"
            exit 1
        fi
        ;;
    windows-x86_64)
        echo "Windows is not supported yet"
        exit 1
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
