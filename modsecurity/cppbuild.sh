#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" modsecurity
    popd
    exit
fi

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

if [[ ! -d "ModSecurity" ]]; then
    git clone https://github.com/SpiderLabs/ModSecurity
    cd ModSecurity
    git checkout v3.0.6
    git submodule init
    git submodule update
else
    cd ModSecurity
fi

case $PLATFORM in
    linux-x86_64)
        sh build.sh
        ./configure --prefix=$INSTALL_PATH
        make
        make install
        ;;
    macosx-x86_64)
        sh build.sh
        sedinplace 's/\\\$rpath/@rpath/g' configure
        ./configure --prefix=$INSTALL_PATH
        make
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
