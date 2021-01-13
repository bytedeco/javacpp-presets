#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" modsecurity
    popd
    exit
fi


sudo apt-get install g++ flex bison curl doxygen libyajl-dev libgeoip-dev libtool dh-autoreconf libcurl4-gnutls-dev libxml2 libpcre++-dev libxml2-dev
mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

if   [ ! -d "ModSecurity" ];
then
  git clone https://github.com/SpiderLabs/ModSecurity
  cd ModSecurity
  git checkout origin/v3/master
  git submodule init
  git submodule update
else
    cd ModSecurity
fi

case $PLATFORM in
    linux-x86)
        sh build.sh
        ./configure --prefix=$INSTALL_PATH
        make
        make install
        ;;
    linux-x86_64)
        sh build.sh
        ./configure --prefix=$INSTALL_PATH
        make
        make install
        ;;
    macosx-x86_64)
        sh build.sh
        ./configure --prefix=$INSTALL_PATH
        make
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..