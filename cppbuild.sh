#!/bin/bash
# Scripts to build and install native C++ libraries
set -eu

which cmake3 &> /dev/null && CMAKE3="cmake3" || CMAKE3="cmake"
[[ -z ${CMAKE:-} ]] && CMAKE=$CMAKE3
[[ -z ${MAKEJ:-} ]] && MAKEJ=4
[[ -z ${OLDCC:-} ]] && OLDCC="gcc"
[[ -z ${OLDCXX:-} ]] && OLDCXX="g++"
[[ -z ${OLDFC:-} ]] && OLDFC="gfortran"

KERNEL=(`uname -s | tr [A-Z] [a-z]`)
ARCH=(`uname -m | tr [A-Z] [a-z]`)
case $KERNEL in
    darwin)
        OS=macosx
        ;;
    mingw32*)
        OS=windows
        KERNEL=windows
        ARCH=x86
        ;;
    mingw64*)
        OS=windows
        KERNEL=windows
        ARCH=x86_64
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
echo "Building for platform \"$PLATFORM\""

if [[ -z ${OPERATION:-} ]]; then
    echo "Usage: ANDROID_NDK=/path/to/android-ndk/ bash cppbuild.sh [-platform <name>] <install | clean> [projects]"
    echo "where possible platform names are: android-arm, android-x86, linux-x86, linux-x86_64, macosx-x86_64, windows-x86, windows-x86_64, etc."
    exit 1
fi

if [[ -z ${ANDROID_NDK:-} ]]; then
    ANDROID_NDK=~/Android/android-ndk/
fi
export ANDROID_NDK
export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.9/"
case $PLATFORM in
    android-x86)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/x86-4.9/prebuilt/$KERNEL-$ARCH/bin/i686-linux-android"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-14/arch-x86/"
        ;;
    *)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/arm-linux-androideabi-4.9/prebuilt/$KERNEL-$ARCH/bin/arm-linux-androideabi"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-14/arch-arm/"
        ;;
esac

TOP_PATH=`pwd`

function download {
    mkdir -p "$TOP_PATH/downloads"
    if [[ ! -e "$TOP_PATH/downloads/$2" ]]; then
        echo "Downloading $1"
        curl -L "$1" -o "$TOP_PATH/downloads/$2" --fail
        DOWNLOADSTATUS=$?
        if [ "$DOWNLOADSTATUS" -eq 28 ]
        then
		echo "Download timed out, waiting 5 minutes then trying again"
		rm "$TOP_PATH/downloads/$2"
		sleep 600
        	curl -L "$1" -o "$TOP_PATH/downloads/$2" --fail
        	if [ $? -ne 0 ]
        	then
			echo "File still could not be downloaded!"
			rm "$TOP_PATH/downloads/$2"
			exit 1
    		fi
        elif [ "$DOWNLOADSTATUS" -ne 0 ]
        then
		echo "File could not be downloaded!"
		rm "$TOP_PATH/downloads/$2"
		exit 1
        fi
    fi
    ln -sf "$TOP_PATH/downloads/$2" "$2"
}

if [[ -z ${PROJECTS:-} ]]; then
    PROJECTS=(opencv ffmpeg flycapture libdc1394 libfreenect libfreenect2 librealsense videoinput artoolkitplus chilitags flandmark hdf5 mkl openblas fftw gsl llvm leptonica tesseract caffe cuda mxnet tensorflow ale liquidfun skia)
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
