#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" helloworld
    popd
    exit
fi

HELLOWORLD_VERSION=master
download https://github.com/matteodg/helloworld/archive/$HELLOWORLD_VERSION.zip helloworld-$HELLOWORLD_VERSION.zip

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`

#git clone https://github.com/matteodg/helloworld.git helloworld-$HELLOWORLD_VERSION
echo "Decompressing archives..."
unzip -o ../helloworld-$HELLOWORLD_VERSION.zip

cd helloworld-$HELLOWORLD_VERSION

case $PLATFORM in
    linux-x86|linux-x86_64|macosx-*|windows-x86|windows-x86_64)
	./autogen.sh
        ./configure --prefix=$INSTALL_PATH
        make
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
