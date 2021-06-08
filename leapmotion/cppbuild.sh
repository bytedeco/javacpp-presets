#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script

if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" leapmotion
    popd
    exit
fi

mkdir -p $PLATFORM
cd $PLATFORM

case $PLATFORM in
    # android-arm)
    #     CC="$ANDROID_CC" CFLAGS="$ANDROID_FLAGS" ./configure --prefix=.. --static
    #     make -j $MAKEJ
    #     make install
    #     ;;
    # android-arm64)
    #     CC="$ANDROID_CC" CFLAGS="$ANDROID_FLAGS" ./configure --prefix=.. --static
    #     make -j $MAKEJ
    #     make install
    #     ;;
    # android-x86)
    #     CC="$ANDROID_CC" CFLAGS="$ANDROID_FLAGS" ./configure --prefix=.. --static
    #     make -j $MAKEJ
    #     make install
    #     ;;
    # android-x86_64)
    #     CC="$ANDROID_CC" CFLAGS="$ANDROID_FLAGS" ./configure --prefix=.. --static
    #     make -j $MAKEJ
    #     make install
    #     ;;
    # linux-x86)
    #     CC="gcc -m32 -fPIC" ./configure --prefix=.. --static
    #     make -j $MAKEJ
    #     make install
    #     ;;
    # linux-x86_64)
    #     CC="gcc -m64 -fPIC" ./configure --prefix=.. --static
    #     make -j $MAKEJ
    #     make install
    #     ;;
    # macosx-x86_64)
    #     ./configure --prefix=.. --static
    #     make -j $MAKEJ
    #     make install
    #     ;;
    # windows-x86)
    #     nmake -f win32/Makefile.msc zlib.lib
    #     mkdir -p ../include ../lib
    #     cp zconf.h zlib.h ../include/
    #     cp zlib.lib ../lib/
    #     ;;
    windows-x86_64)
        LEAPMOTION_VERSION=4.1.0
        EXTENSION=.zip
        FOLDER=LeapDeveloperKit_$LEAPMOTION_VERSION+52211_win
        FILE=$FOLDER$EXTENSION
        cd ../../
        if [ -f $FILE ];then
            echo "$FILE exists, unzipping..."
        else
            echo "$FILE not found. Please download it and place it into this directory."
            exit 1
        fi
        unzip $FILE
        cd cppbuild/$PLATFORM/
        mkdir -p lib
        cp ../../$FOLDER/LeapSDK/lib/x64/LeapC.lib lib/
        cp -r ../../$FOLDER/LeapSDK/include include
        rm -rf ../../$FOLDER
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
