#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" openvino
    popd
    exit
fi

OPENVINO_VERSION=2025.4.1
OPENVINO_BUILD=20426.82bbf0292c5
OPENVINO_PACKAGES_URL=https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION}

mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`
mkdir -p include lib bin

extract_openvino_archive() {
    local archive_platform=$1
    local archive_name=$2
    local archive_dir

    find . -maxdepth 1 -type d -name "openvino_toolkit_*" -exec rm -rf {} \;
    download "${OPENVINO_PACKAGES_URL}/${archive_platform}/${archive_name}" "${archive_name}"

    case "$archive_name" in
        *.zip)
            unzip -q -o "$archive_name"
            ;;
        *.tar.gz|*.tgz)
            tar -xzf "$archive_name"
            ;;
        *)
            echo "Error: Unsupported archive \"$archive_name\""
            exit 1
            ;;
    esac

    archive_dir=$(find . -maxdepth 1 -type d -name "openvino_toolkit_*_${OPENVINO_VERSION}.${OPENVINO_BUILD}_x86_64" | head -n 1)
    if [[ -z "$archive_dir" ]]; then
        echo "Error: OpenVINO archive contents not found in \"$archive_name\""
        exit 1
    fi

    cp -a "$archive_dir/runtime/include/." include/

    case $PLATFORM in
        linux-x86_64)
            cp -a "$archive_dir/runtime/lib/intel64/." lib/
            ;;
        windows-x86_64)
            cp -a "$archive_dir/runtime/bin/intel64/Release/." bin/
            cp -a "$archive_dir/runtime/lib/intel64/Release/." lib/
            ;;
    esac

    rm -rf "$archive_dir"
}

case $PLATFORM in
    linux-x86_64)
        extract_openvino_archive linux openvino_toolkit_ubuntu22_${OPENVINO_VERSION}.${OPENVINO_BUILD}_x86_64.tgz
        pushd lib
        ln -sf $(ls libopenvino.so.* | head -n 1) libopenvino.so
        ln -sf $(ls libopenvino_c.so.* | head -n 1) libopenvino_c.so
        popd
        ;;
    windows-x86_64)
        extract_openvino_archive windows openvino_toolkit_windows_${OPENVINO_VERSION}.${OPENVINO_BUILD}_x86_64.zip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        exit 1
        ;;
esac

cd ..
