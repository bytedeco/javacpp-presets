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
PYTHON_VERSION=310

mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`
mkdir -p include lib bin

download_openvino_wheel() {
    local wheel_platform=$1
    local wheel_file
    local wheel_build=${OPENVINO_BUILD%%.*}

    if [[ "$wheel_platform" == "manylinux2014_x86_64" ]]; then
        python3 -m pip download --no-deps openvino==${OPENVINO_VERSION} -d .
    else
        python3 -m pip download --no-deps --platform "$wheel_platform" --only-binary=:all: \
            --python-version "$PYTHON_VERSION" --implementation cp --abi cp$PYTHON_VERSION \
            openvino==${OPENVINO_VERSION} -d .
    fi
    wheel_file=$(ls openvino-${OPENVINO_VERSION}-${wheel_build}-cp$PYTHON_VERSION-cp$PYTHON_VERSION-$wheel_platform.whl 2>/dev/null \
        || ls openvino-${OPENVINO_VERSION}-*-cp$PYTHON_VERSION-cp$PYTHON_VERSION-$wheel_platform.whl)

    python3 - "$wheel_file" "$INSTALL_PATH" <<'PY'
import os, shutil, sys, zipfile
wheel, install = sys.argv[1:3]
tmp = os.path.join(install, "openvino-wheel")
if os.path.exists(tmp):
    shutil.rmtree(tmp)
os.makedirs(tmp)
with zipfile.ZipFile(wheel) as zf:
    zf.extractall(tmp)
include_src = os.path.join(tmp, "openvino", "include")
libs_src = os.path.join(tmp, "openvino", "libs")
shutil.copytree(include_src, os.path.join(install, "include"), dirs_exist_ok=True)
os.makedirs(os.path.join(install, "lib"), exist_ok=True)
os.makedirs(os.path.join(install, "bin"), exist_ok=True)
for n in os.listdir(libs_src):
    p = os.path.join(libs_src, n)
    if n.endswith((".so", ".so.1", ".so.2", ".so.3", ".so.4", ".so.5", ".so.6", ".so.7", ".so.8", ".so.9")) or ".so." in n:
        shutil.copy2(p, os.path.join(install, "lib", n))
    elif n.endswith(".dll"):
        shutil.copy2(p, os.path.join(install, "bin", n))
    elif n.endswith(".lib"):
        shutil.copy2(p, os.path.join(install, "lib", n))
PY

    rm -rf openvino-wheel
}

case $PLATFORM in
    linux-x86_64)
        download_openvino_wheel manylinux2014_x86_64
        pushd lib
        ln -sf $(ls libopenvino.so.* | head -n 1) libopenvino.so
        ln -sf $(ls libopenvino_c.so.* | head -n 1) libopenvino_c.so
        popd
        ;;
    windows-x86_64)
        download_openvino_wheel win_amd64
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        exit 1
        ;;
esac

cd ..
