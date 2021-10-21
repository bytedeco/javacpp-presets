#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" frovedis
    popd
    exit
fi

case $PLATFORM in
    linux-x86_64)
        if [[ ! -d "/opt/nec/frovedis/" ]]; then
            echo "Please install Frovedis under the default installation directory"
            exit 1
        fi
        mkdir -p ../target/classes
        unzip /opt/nec/frovedis/x86/lib/spark/frovedis_client.jar -d ../target/classes
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
