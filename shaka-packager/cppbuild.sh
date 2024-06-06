#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" shaka-packager
    popd
    exit
fi


# Compilation instructions at https://github.com/shaka-project/shaka-packager/blob/main/docs/source/build_instructions.md


mkdir -p $PLATFORM
cd $PLATFORM

CURRENT_PLATFORM_PATH=$(pwd)


echo "Cloning reposository..."

if [[ ! -d shaka-packager ]]; then
    git clone --recurse-submodules https://github.com/shaka-project/shaka-packager.git

fi
cd shaka-packager
git submodule update --init --recursive
INSTALL_PATH=$(pwd)




case $PLATFORM in
    linux-x86_64)
        cmake -S . -B build -DBUILD_SHARED_LIBS="ON"  -DCMAKE_BUILD_TYPE=Release -G Ninja  -DCMAKE_INSTALL_PREFIX=$CURRENT_PLATFORM_PATH
        cmake --build build/ --config Release --parallel 
        cmake --install build/ --strip --config Release --prefix=$CURRENT_PLATFORM_PATH 
        ;;
    linux-arm64)
        cmake -S . -B build -DBUILD_SHARED_LIBS="ON"  -DCMAKE_BUILD_TYPE=Release -G Ninja  -DCMAKE_INSTALL_PREFIX=$CURRENT_PLATFORM_PATH
        cmake --build build/ --config Release --parallel 
        cmake --install build/ --strip --config Release --prefix=$CURRENT_PLATFORM_PATH
        ;;
    macosx-arm64)
        cmake -S . -B build -DBUILD_SHARED_LIBS="ON"  -DCMAKE_BUILD_TYPE=Release -G Ninja  -DCMAKE_INSTALL_PREFIX=$CURRENT_PLATFORM_PATH
        cmake --build build/ --config Release --parallel 
        cmake --install build/ --strip --config Release --prefix=$CURRENT_PLATFORM_PATH
        ;;
    macosx-x86_64)
        cmake -S . -B build -DBUILD_SHARED_LIBS="ON"  -DCMAKE_BUILD_TYPE=Release -G Ninja  -DCMAKE_INSTALL_PREFIX=$CURRENT_PLATFORM_PATH
        cmake --build build/ --config Release --parallel 
        cmake --install build/ --strip --config Release --prefix=$CURRENT_PLATFORM_PATH
        ;;
    windows-x86_64)
        cmake -S . -B build -DBUILD_SHARED_LIBS="ON"  -DCMAKE_BUILD_TYPE=Debug  -DCMAKE_INSTALL_PREFIX=$CURRENT_PLATFORM_PATH
        cmake --build build/ --config Debug --parallel 
        cmake --install build/ --strip --config Debug --prefix=$CURRENT_PLATFORM_PATH
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..

