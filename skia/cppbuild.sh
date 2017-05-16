#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" skia
    popd
    exit
fi

case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        ;;
    macosx-*)
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

# Must be kept in sync with skia.version in pom.xml
SKIA_VERSION=53d672981d2f4535d61da05befa793a73103c4fd
download "https://github.com/mono/skia/archive/$SKIA_VERSION.zip" "skia-$SKIA_VERSION.zip"

if [ ! -d depot_tools ]; then
    echo "Fetching depot_tools..."
    git clone 'https://chromium.googlesource.com/chromium/tools/depot_tools.git' depot_tools
fi
export PATH="$PWD/depot_tools:$PATH"

mkdir -p "$PLATFORM"
cd "$PLATFORM"
if [ ! -d "skia-$SKIA_VERSION" ]; then
    echo "Decompressing archives..."
    unzip "../skia-$SKIA_VERSION.zip"
fi
cd "skia-$SKIA_VERSION"
python tools/git-sync-deps

bin/gn gen out/Shared --args='is_official_build=false is_debug=false is_component_build=true extra_cflags=["-DSKIA_C_DLL"]'
ninja -C out/Shared

cd ../..
