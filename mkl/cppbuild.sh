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
        if [[ ! -d "/opt/intel/mkl/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /opt/intel/mkl/include/* include/
        cp -a /opt/intel/lib/ia32/* /opt/intel/mkl/lib/ia32/* lib/
        ;;
    linux-x86_64)
        if [[ ! -d "/opt/intel/mkl/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /opt/intel/mkl/include/* include/
        cp -a /opt/intel/lib/intel64/* /opt/intel/mkl/lib/intel64/* lib/
        ;;
    macosx-*)
        if [[ ! -d "/opt/intel/mkl/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /opt/intel/mkl/include/* include/
        cp -a /opt/intel/lib/* /opt/intel/mkl/lib/* lib/
        ;;
    windows-x86)
        if [[ ! -d "/C/Program Files (x86)/IntelSWTools/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /C/Program\ Files\ \(x86\)/IntelSWTools/compilers_and_libraries/windows/mkl/include/* include/
        cp -a /C/Program\ Files\ \(x86\)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/ia32/* lib/
        cp -a /C/Program\ Files\ \(x86\)/IntelSWTools/compilers_and_libraries/windows/redist/ia32/compiler/* bin/
        cp -a /C/Program\ Files\ \(x86\)/IntelSWTools/compilers_and_libraries/windows/redist/ia32/mkl/* bin/
        ;;
    windows-x86_64)
        if [[ ! -d "/C/Program Files (x86)/IntelSWTools/" ]]; then
            echo "Please install MKL under the default installation directory"
            exit 1
        fi
        cp -a /C/Program\ Files\ \(x86\)/IntelSWTools/compilers_and_libraries/windows/mkl/include/* include/
        cp -a /C/Program\ Files\ \(x86\)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64/* lib/
        cp -a /C/Program\ Files\ \(x86\)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/compiler/* bin/
        cp -a /C/Program\ Files\ \(x86\)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/* bin/
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
