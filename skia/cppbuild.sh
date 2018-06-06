#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" skia
    popd
    exit
fi

export TARGET_CPU=
export EXTRA_ARGS=
case $PLATFORM in
    ios-arm64)
        export CC="$(xcrun --sdk iphoneos --find clang) -isysroot $(xcrun --sdk iphoneos --show-sdk-path) -arch arm64"
        export CXX="$(xcrun --sdk iphoneos --find clang++) -isysroot $(xcrun --sdk iphoneos --show-sdk-path) -arch arm64"
        export EXTRA_ARGS='target_os="ios"'
        export TARGET_CPU="arm64"
        ;;
    ios-x86_64)
        export CC="$(xcrun --sdk iphonesimulator --find clang) -isysroot $(xcrun --sdk iphonesimulator --show-sdk-path) -arch x86_64"
        export CXX="$(xcrun --sdk iphonesimulator --find clang++) -isysroot $(xcrun --sdk iphonesimulator --show-sdk-path) -arch x86_64"
        export EXTRA_ARGS='target_os="ios"'
        export TARGET_CPU="x64"
        ;;
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
# Upstream doesn't appear to build ios with is_component_build=true
patch -p0 skia-$SKIA_VERSION/gn/BUILD.gn <<-PATCH
696c696
<     if (is_mac) {
---
>     if (is_mac || is_ios) {
PATCH
export PATH="$PWD/depot_tools:$PATH"

cd skia-$SKIA_VERSION
# Work around the disappearance of https://skia.googlesource.com/third_party/libjpeg-turbo.git
patch -Np1 < ../../../skia.patch || true
python2 tools/git-sync-deps
cp third_party/libjpeg-turbo/* third_party/externals/libjpeg-turbo/

if [[ $PLATFORM == ios* ]]; then
    sed -i="" s/thread_local//g tools/ok.cpp
    sed -i="" /SRC_SK_XFERMODE_MODE/d tests/CTest.cpp
    bin/gn gen out/Static --script-executable=python2 --args="target_cpu=\"$TARGET_CPU\" is_official_build=false is_debug=false extra_cflags=[\"-g0\"] $EXTRA_ARGS"
    ninja -C out/Static
else
    bin/gn gen out/Shared --script-executable=python2 --args="target_cpu=\"$TARGET_CPU\" is_official_build=false is_debug=false is_component_build=true extra_cflags=[\"-g0\", \"-DSKIA_C_DLL\"] $EXTRA_ARGS"
    ninja -C out/Shared
fi

cd ../..
