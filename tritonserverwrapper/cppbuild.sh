#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tritonserverwrapper
    popd
    exit
fi

case $PLATFORM in
    linux-arm64)
        if [[ ! -d "/opt/tritonserver/lib/" ]] && [[ ! -d "/opt/tritonserver/developer_tools/" ]]; then
            echo "Please make sure library and include files exist"
            exit 1
        fi
        ;;
    linux-x86_64)
        if [[ ! -d "/opt/tritonserver/lib/" ]] && [[ ! -d "/opt/tritonserver/developer_tools/" ]]; then
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
