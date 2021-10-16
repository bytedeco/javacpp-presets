#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" veoffload
    popd
    exit
fi

case $PLATFORM in
    linux-x86_64)
        if [[ ! -d "/opt/nec/ve/veos/" ]]; then
            echo "Please install veoffload under the default installation directory"
            exit 1
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
