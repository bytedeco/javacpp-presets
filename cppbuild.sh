#!/bin/bash
# Scripts to build and install native C++ libraries
set -eux

if ! NCPUS=$(grep -c ^proc /proc/cpuinfo); then
    NCPUS=4
fi
if (( NCPUS > 4 )); then NCPUS=4; fi
export NCPUS

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

if [[ -z ${OPERATION:-} ]]; then
    echo "Usage: ANDROID_NDK=/path/to/android-ndk/ bash cppbuild.sh [-platform <name>] <install | clean> [projects]"
    echo "where possible platform names are: android-arm, android-x86, linux-x86, linux-x86_64, macosx-x86_64, windows-x86, windows-x86_64, etc."
    exit 1
fi

if [[ -z ${ANDROID_NDK:-} ]]; then
    ANDROID_NDK=~/projects/android/android-ndk/
fi
export ANDROID_NDK
export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.6/"
case $PLATFORM in
    android-x86)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/x86-4.6/prebuilt/$KERNEL-$ARCH/bin/i686-linux-android"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-9/arch-x86/"
        ;;
    *)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/arm-linux-androideabi-4.6/prebuilt/$KERNEL-$ARCH/bin/arm-linux-androideabi"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-9/arch-arm/"
        ;;
esac

export CACHEDIR=$(pwd)/.cache
mkdir -p "$CACHEDIR"

function download {
    local url="$1"
    local destfile="$2"
    local cachefile="$CACHEDIR/${destfile##*/}"
    local tmpfile="$cachefile.tmp"
    if ! test -e "$cachefile"; then
        rm -f "$tmpfile"
        if curl -L -o "$tmpfile" "$url"; then
            mv -f "$tmpfile" "$cachefile"
        else
            echo "failed to retrieve $url" >&2
            exit 1
        fi
    fi
    cp -vf "$cachefile" "$destfile"
    return 0
}

function getgit {
    local project="$1"
    local tagname="$2"
    local destfile="$3"
    download "https://codeload.github.com/$project/tar.gz/$tagname" "$destfile"
}

if [[ -z ${PROJECTS:-} ]]; then
    PROJECTS=(opencv ffmpeg flycapture libdc1394 libfreenect videoinput artoolkitplus flandmark fftw gsl llvm leptonica tesseract)
fi

for PROJECT in ${PROJECTS[@]}; do
    case $OPERATION in
        install)
            if [[ ! -d $PROJECT ]]; then
                echo "Warning: Project \"$PROJECT\" not found"
            else
                echo "Installing \"$PROJECT\""
                mkdir -p $PROJECT/cppbuild
                pushd $PROJECT/cppbuild
                source ../cppbuild.sh
                popd
            fi
            ;;
        clean)
            echo "Cleaning \"$PROJECT\""
            rm -Rf $PROJECT/cppbuild
            ;;
    esac
done
