#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" systems
    popd
    exit
fi

mkdir -p $PLATFORM
cd $PLATFORM

INCLUDE_PATH="/usr/include/"
if [[ ! -d "$INCLUDE_PATH" ]] && [[ ! -d "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/" ]] &&
        [[ ! -d "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/" ]]; then
    echo "Please install system development files under the default installation directory"
    exit 1
fi

case $PLATFORM in
    linux-armhf)
        CROSS_INCLUDE_PATH=$(echo | arm-linux-gnueabihf-g++ -E -v - 2>&1 | grep -o '^ .*/usr/include' | tail -1 | xargs)
        if [[ -d "$CROSS_INCLUDE_PATH" ]]; then
            INCLUDE_PATH="$CROSS_INCLUDE_PATH"
        fi
        touch cpuid.h
        ln -sf "$INCLUDE_PATH"
        ln -sf "$INCLUDE_PATH/arm-linux-gnueabihf"
        ;;
    linux-arm64)
        touch cpuid.h
        ln -sf "$INCLUDE_PATH"
        ln -sf "$INCLUDE_PATH/aarch64-linux-gnu"
        ;;
    linux-ppc64le)
        touch cpuid.h
        ln -sf "$INCLUDE_PATH"
        ln -sf "$INCLUDE_PATH/powerpc64le-linux-gnu"
        ;;
    linux-x86*)
        CPUID_PATH=$(echo '#include <cpuid.h>' | g++ -M -E - | grep -o ' .*cpuid.h' | xargs)
        ln -sf "$CPUID_PATH"
        ln -sf "$INCLUDE_PATH"
        ln -sf "$INCLUDE_PATH/i386-linux-gnu"
        ln -sf "$INCLUDE_PATH/x86_64-linux-gnu"
        ;;
    macosx-*)
        CPUID_PATH=$(echo '#include <cpuid.h>' | clang++ -M -E - | grep -o ' .*cpuid.h' | xargs)
        ln -sf "$CPUID_PATH"
        ln -sf "$INCLUDE_PATH"
        ;;
    windows-*)
        if [[ ! -d "/C/Program Files (x86)/Windows Kits/" ]]; then
            echo "Please install the Windows SDK under the default installation directory"
            exit 1
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..

