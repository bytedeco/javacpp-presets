#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" systems
    popd
    exit
fi

mkdir -p $PLATFORM/include
cd $PLATFORM/include

case $PLATFORM in
    linux-x86*)
        if [[ ! -d "/usr/include/" ]]; then
            echo "Please install system development files under the default installation directory"
            exit 1
        fi
        CPUID_PATH=$(echo '#include <cpuid.h>' | g++ -M -E - | grep -o ' .*cpuid.h')
        ln -sf $CPUID_PATH
        ;;
    macosx-*)
        if [[ ! -d "/usr/include/" ]]; then
            echo "Please install system development files under the default installation directory"
            exit 1
        fi
        CPUID_PATH=$(echo '#include <cpuid.h>' | clang++ -M -E - | grep -o ' .*cpuid.h')
        ln -sf $CPUID_PATH
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

