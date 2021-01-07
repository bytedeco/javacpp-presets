#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" mkl
    popd
    exit
fi

mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`
mkdir -p include lib bin

case $PLATFORM in
    linux-x86)
        if [[ ! -d "/opt/intel/oneapi/mkl/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /opt/intel/oneapi/mkl/latest/include/* include/
        cp -a /opt/intel/oneapi/mkl/latest/lib/ia32/* /opt/intel/oneapi/compiler/latest/linux/compiler/lib/ia32_lin/* lib/
        ;;
    linux-x86_64)
        if [[ ! -d "/opt/intel/oneapi/mkl/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /opt/intel/oneapi/mkl/latest/include/* include/
        cp -a /opt/intel/oneapi/mkl/latest/lib/intel64/* /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/* lib/
        ;;
    macosx-*)
        if [[ ! -d "/opt/intel/oneapi/mkl/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /opt/intel/oneapi/mkl/latest/include/* include/
        cp -a /opt/intel/oneapi/mkl/latest/lib/* /opt/intel/oneapi/compiler/latest/mac/compiler/lib/* lib/
        ;;
    windows-x86)
        if [[ ! -d "/C/Program Files (x86)/Intel/oneAPI/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/include/* include/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/lib/ia32/* lib/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/redist/ia32/* bin/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/compiler/latest/windows/redist/ia32_win/compiler/* bin/
        ;;
    windows-x86_64)
        if [[ ! -d "/C/Program Files (x86)/Intel/oneAPI/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/include/* include/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/lib/intel64/* lib/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/redist/intel64/* bin/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/compiler/latest/windows/redist/intel64_win/compiler/* bin/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
