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
        export CC="clang"
        export CXX="clang++"
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

# Must be kept in sync with skia.version in pom.xml
SKIA_VERSION=2.80.3
download https://chromium.googlesource.com/chromium/tools/depot_tools.git/+archive/master.tar.gz depot_tools.tar.gz
download https://github.com/mono/skia/archive/v$SKIA_VERSION.tar.gz skia-$SKIA_VERSION.tar.gz

mkdir -p "$PLATFORM"
cd "$PLATFORM"

echo "Decompressing archives..."
mkdir -p depot_tools
tar --totals -xzf ../depot_tools.tar.gz -C depot_tools
tar --totals -xzf ../skia-$SKIA_VERSION.tar.gz

sedinplace '/for thread in threads:/d' skia-$SKIA_VERSION/tools/git-sync-deps
sedinplace 's/"HAVE_MEMMOVE"/"HAVE_MEMMOVE", "XML_DEV_URANDOM"/g' skia-$SKIA_VERSION/third_party/expat/BUILD.gn
#sedinplace '/sources = tests_sources/,/}/d' skia-$SKIA_VERSION/BUILD.gn
sedinplace /-ffp-contract=fast/d skia-$SKIA_VERSION/BUILD.gn
sedinplace /-march=haswell/d skia-$SKIA_VERSION/BUILD.gn
sedinplace /-Werror/d skia-$SKIA_VERSION/gn/BUILD.gn
export PATH="$PWD/depot_tools:$PATH"

cd skia-$SKIA_VERSION
python tools/git-sync-deps

if [[ $PLATFORM == ios* ]]; then
    bin/gn gen out/Static --args="target_cxx=\"$CXX\" target_cpu=\"$TARGET_CPU\" is_official_build=false is_debug=false extra_cflags=[\"-g0\", \"-I../../third_party/externals/freetype/include/\"] $EXTRA_ARGS"
    ninja -C out/Static skia
else
    bin/gn gen out/Shared --args="target_cxx=\"$CXX\" target_cpu=\"$TARGET_CPU\" is_official_build=false is_debug=false is_component_build=true extra_cflags=[\"-g0\", \"-I../../third_party/externals/freetype/include/\", \"-DSKIA_C_DLL\"] $EXTRA_ARGS"
    ninja -C out/Shared skia
fi

cd ../..
