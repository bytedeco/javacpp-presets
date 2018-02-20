#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" vimba
    popd
    exit
fi

# Download Linux drivers
TIS_PATH="$(pwd)/tiscamera"

# Helps debugging...
if ! [ -d "$TIS_PATH" ]; then
rm -rf "$TIS_PATH"
git clone --recursive https://github.com/TheImagingSource/tiscamera.git "$TIS_PATH"

# Apply patch
echo "Patching TIS source code with ../tiscamera.patch"
patch -s -p0 < ../tiscamera.patch

INSTALL_PATH="$(pwd)/$PLATFORM$EXTENSION"
INCLUDE_PATH="$INSTALL_PATH/include"
LIB_PATH="$INSTALL_PATH/lib"

mkdir -p "${INSTALL_PATH}"

if [ $PLATFORM == linux-x86* ]
then

    cd "$TIS_PATH"
    mkdir -p build
    cd build

    # With ARAVIS:
    #cmake -DBUILD_ARAVIS=ON -DBUILD_GST_1_0=ON -DBUILD_TOOLS=ON -DBUILD_V4L2=ON -DCMAKE_INSTALL_PREFIX=/usr ..
    # Without ARAVIS
    cmake -DBUILD_ARAVIS=OFF -DBUILD_GST_1_0=ON -DBUILD_TOOLS=OFF -DBUILD_V4L2=OFF -DCMAKE_INSTALL_PREFIX=/usr ..

    make -j4 VERBOSE=1

    rm -rf $INCLUDE_PATH
    mkdir -p $INCLUDE_PATH
    cp $TIS_PATH/src/*.h "$INCLUDE_PATH/"
    cp $TIS_PATH/src/*.cpp "$INCLUDE_PATH/"

    mkdir -p $INCLUDE_PATH/algorithms
    cp -r $TIS_PATH/src/algorithms/*.h "$INCLUDE_PATH/algorithms/"

    mkdir -p $INCLUDE_PATH/gstreamer-1.0
    cp -r $TIS_PATH/src/gstreamer-1.0/*.h "$INCLUDE_PATH/gstreamer-1.0/"

    mkdir -p $INCLUDE_PATH/gobject
    cp -r $TIS_PATH/src/gobject/*.h "$INCLUDE_PATH/gobject/"

    rm -rf $LIB_PATH
    mkdir -p $LIB_PATH/
    # TODO why doesn't this work???
    #cp $TIS_PATH/build/src/libtcam.so* "$LIB_PATH/"
    #cp $TIS_PATH/build/src/gobject/libtcamprop.so* "$LIB_PATH/"

else
    echo "Error: Platform \"$PLATFORM\" is not supported"
fi
fi