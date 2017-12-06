#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" skia
    popd
    exit
fi

export TARGET_CPU=
case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        export TARGET_CPU="x86"
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        export TARGET_CPU="x64"
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
download "https://chromium.googlesource.com/chromium/tools/depot_tools.git/+archive/master.tar.gz" "depot_tools.tar.gz"
download "https://github.com/mono/skia/archive/$SKIA_VERSION.tar.gz" "skia-$SKIA_VERSION.tar.gz"

mkdir -p "$PLATFORM"
cd "$PLATFORM"

echo "Decompressing archives..."
mkdir -p depot_tools
tar --totals -xzf ../depot_tools.tar.gz -C depot_tools
tar --totals -xzf ../skia-$SKIA_VERSION.tar.gz

sed -i="" /-Werror/d skia-$SKIA_VERSION/gn/BUILD.gn
export PATH="$PWD/depot_tools:$PATH"

cd skia-$SKIA_VERSION
python tools/git-sync-deps

bin/gn gen out/Shared --args="target_cpu=\"$TARGET_CPU\" is_official_build=false is_debug=false is_component_build=true extra_cflags=[\"-DSKIA_C_DLL\"]"
ninja -C out/Shared

cd ../..
