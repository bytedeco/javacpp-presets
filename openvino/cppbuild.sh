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

archive_name=
archive_platform=
archive_dir_pattern="openvino_toolkit_*_${OPENVINO_VERSION}.${OPENVINO_BUILD}_x86_64"
runtime_lib_path=
runtime_bin_path=
tbb_lib_path=runtime/3rdparty/tbb/lib
tbb_bin_path=
tbb_lib_files=()
tbb_bin_files=()
soname_links=()

case $PLATFORM in
    linux-x86_64)
        archive_name=openvino_toolkit_ubuntu22_${OPENVINO_VERSION}.${OPENVINO_BUILD}_x86_64.tgz
        archive_platform=linux
        runtime_lib_path=runtime/lib/intel64
        tbb_lib_files=(libhwloc.so* libtbb.so* libtbbbind_2_5.so* libtbbmalloc.so* libtbbmalloc_proxy.so*)
        soname_links=(libopenvino.so libopenvino_c.so libopenvino_ir_frontend.so)
        ;;
    windows-x86_64)
        archive_name=openvino_toolkit_windows_${OPENVINO_VERSION}.${OPENVINO_BUILD}_x86_64.zip
        archive_platform=windows
        runtime_lib_path=runtime/lib/intel64/Release
        runtime_bin_path=runtime/bin/intel64/Release
        tbb_bin_path=runtime/3rdparty/tbb/bin
        tbb_lib_files=(tbb12.lib tbbbind_2_5.lib tbbmalloc.lib tbbmalloc_proxy.lib)
        tbb_bin_files=(tbb12.dll tbbbind_2_5.dll tbbmalloc.dll tbbmalloc_proxy.dll)
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        exit 1
        ;;
esac

find . -maxdepth 1 -type d -name "openvino_toolkit_*" -exec rm -rf {} \;
download "${OPENVINO_PACKAGES_URL}/${archive_platform}/${archive_name}" "${archive_name}"

case "$archive_name" in
    *.tgz)
        tar -xzf "$archive_name"
        ;;
    *.zip)
        unzip -q -o "$archive_name"
        ;;
    *)
        echo "Error: Unsupported archive \"$archive_name\""
        exit 1
        ;;
esac

archive_dir=$(find . -maxdepth 1 -type d -name "${archive_dir_pattern}" | head -n 1)
if [[ -z "$archive_dir" ]]; then
    echo "Error: OpenVINO archive contents not found in \"$archive_name\""
    exit 1
fi

cp -a "$archive_dir/runtime/include/." include/
cp -a "$archive_dir/${runtime_lib_path}/." lib/

if [[ -n "$runtime_bin_path" ]]; then
    cp -a "$archive_dir/${runtime_bin_path}/." bin/
fi

mkdir -p lib/tbb
cp -a "$archive_dir/${tbb_lib_path}/." lib/tbb/

if [[ -n "$tbb_bin_path" ]]; then
    mkdir -p bin/tbb
    cp -a "$archive_dir/${tbb_bin_path}/." bin/tbb/
fi

shopt -s nullglob
for file in "${tbb_lib_files[@]}"; do
    for source_file in "$archive_dir"/${tbb_lib_path}/$file; do
        cp -a "$source_file" lib/
    done
done

for file in "${tbb_bin_files[@]}"; do
    cp -a "$archive_dir/${tbb_bin_path}/$file" bin/
done
shopt -u nullglob

rm -rf "$archive_dir"

if [[ ${#soname_links[@]} -gt 0 ]]; then
    pushd lib
    for link_name in "${soname_links[@]}"; do
        link_target=$(compgen -G "${link_name}.*" | head -n 1)
        if [[ -n "$link_target" ]]; then
            ln -sf "$(basename "$link_target")" "$link_name"
        fi
    done
    popd
fi

cd ..
