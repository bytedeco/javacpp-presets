#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" videoinput
    popd
    exit
fi

VIDEOINPUT_VERSION=master
download https://github.com/ofTheo/videoInput/archive/$VIDEOINPUT_VERSION.zip videoInput-$VIDEOINPUT_VERSION.zip
unzip -o videoInput-$VIDEOINPUT_VERSION.zip
patch -Np0 < ../videoInput-$VIDEOINPUT_VERSION.patch || true

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include
cp -r ../videoInput-$VIDEOINPUT_VERSION/videoInputSrcAndDemos/libs/videoInput/* include
cp -r ../videoInput-$VIDEOINPUT_VERSION/videoInputSrcAndDemos/libs/DShow/Include/* include
cd ..
