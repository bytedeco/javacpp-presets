#!/bin/sh
if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

FLANDMARK_VERSION=a0981a3b09cc5534255dc1dcdae2179097231bdd
download https://github.com/uricamic/flandmark/archive/$FLANDMARK_VERSION.zip flandmark.zip

unzip -o flandmark.zip -d C:/flandmark/sources
