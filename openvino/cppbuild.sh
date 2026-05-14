#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" openvino
    popd
    exit
fi

OPENVINO_VERSION=2026.1.0
OPENVINO_BUILD=21367.63e31528c62
OPENVINO_PACKAGES_URL=https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.1

mkdir -p "$PLATFORM"
cd "$PLATFORM"

archive_name=
archive_platform=
archive_root=

case $PLATFORM in
    linux-x86_64)
        archive_name=openvino_toolkit_ubuntu22_${OPENVINO_VERSION}.${OPENVINO_BUILD}_x86_64.tgz
        archive_platform=linux
        archive_root=${archive_name%.tgz}
        ;;
    windows-x86_64)
        archive_name=openvino_toolkit_windows_${OPENVINO_VERSION}.${OPENVINO_BUILD}_x86_64.zip
        archive_platform=windows
        archive_root=${archive_name%.zip}
        ;;
    macosx-arm64)
        archive_name=openvino_toolkit_macos_12_6_${OPENVINO_VERSION}.${OPENVINO_BUILD}_arm64.tgz
        archive_platform=macos
        archive_root=${archive_name%.tgz}
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        exit 1
        ;;
esac

download "${OPENVINO_PACKAGES_URL}/${archive_platform}/${archive_name}" "${archive_name}"

case "$archive_name" in
    *.tgz)
        tar -xzf "$archive_name" --strip-components=1 "${archive_root}/runtime"
        ;;
    *.zip)
        unzip -q -o "$archive_name"
        if [[ -d "$archive_root/runtime" ]]; then
            mv "$archive_root/runtime" .
            rm -rf "$archive_root"
        fi
        ;;
    *)
        echo "Error: Unsupported archive \"$archive_name\""
        exit 1
        ;;
esac

cd ..
