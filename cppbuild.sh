#!/bin/bash
# Scripts to build and install native C++ libraries

KERNEL=(`uname -s | tr [A-Z] [a-z]`)
ARCH=(`uname -m | tr [A-Z] [a-z]`)
case $KERNEL in
    darwin)
        OS=macosx
        ;;
    mingw*)
        OS=windows
        KERNEL=windows
        if [[ $TARGET_CPU == "x64" ]]; then
            ARCH=x86_64
        fi
        ;;
    *)
        OS=$KERNEL
        ;;
esac
case $ARCH in
    arm*)
        ARCH=arm
        ;;
    i386|i486|i586|i686)
        ARCH=x86
        ;;
    amd64|x86-64)
        ARCH=x86_64
        ;;
esac
PLATFORM=$OS-$ARCH
echo "Detected platform \"$PLATFORM\""

while [[ $# > 0 ]]; do
    case "$1" in
        -platform)
            shift
            PLATFORM="$1"
            ;;
        install)
            OPERATION=install
            ;;
        clean)
            OPERATION=clean
            ;;
        *)
            PROJECTS+=("$1")
            ;;
    esac
    shift
done
echo "Targeting platform \"$PLATFORM\""

if [[ -z "$OPERATION" ]]; then
    echo "Usage: ANDROID_NDK=/path/to/android-ndk-r9d/ bash cppbuild.sh [-platform <name>] [<install | clean>] [projects]"
    echo "where platform includes: android-arm, linux-x86, linux-x86_64, macosx-x86_64, windows-x86, windows-x86_64, etc."
    exit 1
fi

if [[ -z "$ANDROID_NDK" ]]; then
    ANDROID_NDK=~/projects/android/android-ndk-r9d/
fi
case $PLATFORM in
    android-x86)
        ANDROID_TOOLCHAIN="toolchains/x86-4.6/prebuilt/$KERNEL-$ARCH/bin/i686-linux-android"
        ;;
    *)
        ANDROID_TOOLCHAIN="toolchains/arm-linux-androideabi-4.6/prebuilt/$KERNEL-$ARCH/bin/arm-linux-androideabi"
        ;;
esac
export ANDROID_BIN="$ANDROID_NDK/$ANDROID_TOOLCHAIN"
export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.6/"
export ANDROID_ROOT="$ANDROID_NDK/platforms/android-9/arch-arm/"

function download {
    COMMAND="curl -C - -L $1 -o $2"
    echo "$COMMAND"
    $COMMAND
}

if [[ ${#PROJECTS[@]} -eq 0 ]]; then
    PROJECTS=(opencv ffmpeg flycapture libdc1394 libfreenect videoinput artoolkitplus)
fi

for PROJECT in ${PROJECTS[@]}; do
    case $OPERATION in
        install)
            echo "Installing $PROJECT"
            mkdir $PROJECT/cppbuild
            cd $PROJECT/cppbuild
            source ../cppbuild.sh
            cd ../..
            ;;
        clean)
            echo "Cleaning $PROJECT"
            rm -Rf $PROJECT/cppbuild
            ;;
    esac
done
