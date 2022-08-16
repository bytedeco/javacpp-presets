#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libraw
    popd
    exit
fi

LIBRAW_VERSION=0.20.2
download https://www.libraw.org/data/LibRaw-$LIBRAW_VERSION-Win64.zip LibRaw-$LIBRAW_VERSION-Win64.zip
unzip -o LibRaw-$LIBRAW_VERSION-Win64.zip
