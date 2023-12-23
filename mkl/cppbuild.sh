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
        cp -a /opt/intel/oneapi/mkl/latest/lib32/* /opt/intel/oneapi/compiler/latest/lib32/* lib/
        ;;
    linux-x86_64)
        if [[ ! -d "/opt/intel/oneapi/mkl/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /opt/intel/oneapi/mkl/latest/include/* include/
        cp -a /opt/intel/oneapi/mkl/latest/lib/* /opt/intel/oneapi/compiler/latest/lib/* lib/
        ;;
    macosx-*)
        if [[ ! -d "/opt/intel/oneapi/mkl/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /opt/intel/oneapi/mkl/latest/include/* include/
        cp -a /opt/intel/oneapi/mkl/latest/lib/* /opt/intel/oneapi/compiler/latest/lib/* lib/
        ;;
    windows-x86)
        if [[ ! -d "/C/Program Files (x86)/Intel/oneAPI/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/include/* include/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/lib32/* lib/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/bin32/* bin/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/compiler/latest/bin32/* bin/
        ;;
    windows-x86_64)
        if [[ ! -d "/C/Program Files (x86)/Intel/oneAPI/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/include/* include/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/lib/* lib/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/mkl/latest/bin/* bin/
        cp -a /C/Program\ Files\ \(x86\)/Intel/oneAPI/compiler/latest/bin/* bin/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

rm include/ia32 include/intel64 lib/ia32 lib/intel64 lib/locale bin/ia32 bin/intel64 bin/locale | true
