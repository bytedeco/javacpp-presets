#!/bin/bash -e
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" cudf
    popd
    exit
fi

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

RED='\033[1;31m'
NC='\033[0m' # No Color

CUDF_VERSION=0.14.0a200328

case $PLATFORM in
    linux-x86_64)
        if [[ ! -d $INSTALL_PATH/lib ]]; then
            conda create -y -p $INSTALL_PATH -c conda-forge -c rapidsai-nightly/linux-64 libcudf=$CUDF_VERSION libgcc
            conda clean -y --all
        fi
        ;;
    *)
        echo -e "[${RED}ERROR${NC}] Platform \"$PLATFORM\" is not supported"
        ;;
esac
