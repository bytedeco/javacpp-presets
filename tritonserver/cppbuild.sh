#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" tritonserver
    popd
    exit
fi

if [[ ! -f "/opt/tritonserver/include/triton/developer_tools/generic_server_wrapper.h" ]] && [[ ! -f "/opt/tritonserver/lib/libtritondevelopertoolsserver.so" ]] && [ ${INCLUDE_DEVELOPER_TOOLS_SERVER} -ne 1 ]; then
    TOOLS_BRANCH=${TOOLS_BRANCH:="https://github.com/triton-inference-server/developer_tools.git"}
    TOOLS_BRANCH_TAG=${TOOLS_BRANCH_TAG:="main"}
    TRITON_HOME="/opt/tritonserver"
    BUILD_HOME="$PWD"/tritonbuild
    mkdir -p ${BUILD_HOME} && cd ${BUILD_HOME}
    git clone --single-branch --depth=1 -b ${TOOLS_BRANCH_TAG} ${TOOLS_BRANCH}
    cd developer_tools/server
    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BUILD_TEST=ON -DTRITON_ENABLE_EXAMPLES=ON -DTRITON_BUILD_STATIC_LIBRARY=OFF ..
    make -j"$(grep -c ^processor /proc/cpuinfo)" install
    # Copy dynamic library to triton home
    cp ${BUILD_HOME}/developer_tools/server/build/install/lib/libtritondevelopertoolsserver.so ${TRITON_HOME}/lib/.

    mkdir -p ${TRITON_HOME}/include/triton/developer_tools/src
    cp ${BUILD_HOME}/developer_tools/server/include/triton/developer_tools/common.h ${TRITON_HOME}/include/triton/developer_tools/.
    cp ${BUILD_HOME}/developer_tools/server/include/triton/developer_tools/generic_server_wrapper.h ${TRITON_HOME}/include/triton/developer_tools/.
    cp ${BUILD_HOME}/developer_tools/server/src/infer_requested_output.h ${TRITON_HOME}/include/triton/developer_tools/src/.
    cp ${BUILD_HOME}/developer_tools/server/src/tracer.h ${TRITON_HOME}/include/triton/developer_tools/src/
    cd ${BUILD_HOME}/..
    rm -r ${BUILD_HOME}
fi

case $PLATFORM in
    linux-arm64)
        if [[ ! -f "/opt/tritonserver/include/triton/core/tritonserver.h" ]] && [[ ! -d "/opt/tritonserver/lib/" ]]; then
            echo "Please make sure library and include files exist"
            exit 1
        fi
        ;;
    linux-x86_64)
        if [[ ! -f "/opt/tritonserver/include/triton/core/tritonserver.h" ]] && [[ ! -d "/opt/tritonserver/lib/" ]]; then
            echo "Please make sure library and include files exist"
            exit 1
        fi
        ;;
    windows-x86_64)
        echo "Windows is not supported yet"
        exit 1
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
