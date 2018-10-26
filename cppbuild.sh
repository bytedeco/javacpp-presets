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
    aarch64*)
        ARCH=arm64
        ;;
    i386|i486|i586|i686)
        ARCH=x86
        ;;
    amd64|x86-64)
        ARCH=x86_64
        ;;
esac
PLATFORM=$OS-$ARCH
EXTENSION=
echo "Detected platform \"$PLATFORM\""

while [[ $# > 0 ]]; do
    case "$1" in
        -platform=*)
            PLATFORM="${1#-platform=}"
            ;;
        -platform)
            shift
            PLATFORM="$1"
            ;;
        -extension=*)
            EXTENSION="${1#-extension=}"
            ;;
        -extension)
            shift
            EXTENSION="$1"
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

echo -n "Building for platform \"$PLATFORM\""
if [[ -n "$EXTENSION" ]]; then
    echo -n " with extension \"$EXTENSION\""
fi
echo

if [[ -z ${OPERATION:-} ]]; then
    echo "Usage: ANDROID_NDK=/path/to/android-ndk/ bash cppbuild.sh [-platform <name>] [-extension <name>] <install | clean> [projects]"
    echo "where possible platform names are: android-arm, android-x86, linux-x86, linux-x86_64, macosx-x86_64, windows-x86, windows-x86_64, etc."
    exit 1
fi

if [[ -z ${ANDROID_NDK:-} ]]; then
    ANDROID_NDK=~/Android/android-ndk/
fi
export ANDROID_NDK
export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.9/"
case $PLATFORM in
    android-arm)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/arm-linux-androideabi-4.9/prebuilt/$KERNEL-$ARCH/bin/arm-linux-androideabi"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-14/arch-arm/"
        export ANDROID_FLAGS="--sysroot=$ANDROID_ROOT -isystem $ANDROID_NDK/sysroot/usr/include/ -isystem $ANDROID_NDK/sysroot/usr/include/arm-linux-androideabi/ -isystem $ANDROID_CPP/include/ -isystem $ANDROID_CPP/include/backward/ -isystem $ANDROID_CPP/libs/armeabi-v7a/include/ -isystem $ANDROID_NDK/sources/android/cpufeatures/ -D__ANDROID_API__=14 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300 -z text -Wno-attributes -Wl,--fix-cortex-a8 -L$ANDROID_CPP/libs/armeabi-v7a/ -L$ANDROID_ROOT/usr/lib/ -Wl,--no-undefined"
        ;;
    android-arm64)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/aarch64-linux-android-4.9/prebuilt/$KERNEL-$ARCH/bin/aarch64-linux-android"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-21/arch-arm64/"
        export ANDROID_FLAGS="--sysroot=$ANDROID_ROOT -isystem $ANDROID_NDK/sysroot/usr/include/ -isystem $ANDROID_NDK/sysroot/usr/include/aarch64-linux-android/ -isystem $ANDROID_CPP/include/ -isystem $ANDROID_CPP/include/backward/ -isystem $ANDROID_CPP/libs/arm64-v8a/include/ -isystem $ANDROID_NDK/sources/android/cpufeatures/ -D__ANDROID_API__=21 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv8-a -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300 -z text -Wno-attributes -L$ANDROID_CPP/libs/arm64-v8a/ -L$ANDROID_ROOT/usr/lib/ -Wl,--no-undefined"
        ;;
    android-x86)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/x86-4.9/prebuilt/$KERNEL-$ARCH/bin/i686-linux-android"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-14/arch-x86/"
        export ANDROID_FLAGS="--sysroot=$ANDROID_ROOT -isystem $ANDROID_NDK/sysroot/usr/include/ -isystem $ANDROID_NDK/sysroot/usr/include/i686-linux-android/ -isystem $ANDROID_CPP/include/ -isystem $ANDROID_CPP/include/backward/ -isystem $ANDROID_CPP/libs/x86/include/ -isystem $ANDROID_NDK/sources/android/cpufeatures/ -D__ANDROID_API__=14 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=i686 -mtune=atom -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300 -z text -Wno-attributes -L$ANDROID_CPP/libs/x86/ -L$ANDROID_ROOT/usr/lib/ -Wl,--no-undefined"
        ;;
    android-x86_64)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/x86_64-4.9/prebuilt/$KERNEL-$ARCH/bin/x86_64-linux-android"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-21/arch-x86_64/"
        export ANDROID_FLAGS="--sysroot=$ANDROID_ROOT -isystem $ANDROID_NDK/sysroot/usr/include/ -isystem $ANDROID_NDK/sysroot/usr/include/x86_64-linux-android/ -isystem $ANDROID_CPP/include/ -isystem $ANDROID_CPP/include/backward/ -isystem $ANDROID_CPP/libs/x86_64/include/ -isystem $ANDROID_NDK/sources/android/cpufeatures/ -D__ANDROID_API__=21 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=x86-64 -mtune=atom -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300 -z text -Wno-attributes -L$ANDROID_CPP/libs/x86_64/ -L$ANDROID_ROOT/usr/lib64/ -Wl,--no-undefined"
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

function sedinplace {
    if ! sed --version 2>&1 | grep -i gnu > /dev/null; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

if [[ -z ${PROJECTS:-} ]]; then
    PROJECTS=(opencv ffmpeg flycapture spinnaker libdc1394 libfreenect libfreenect2 librealsense videoinput artoolkitplus chilitags flandmark hdf5 mkl mkl-dnn openblas arpack-ng cminpack fftw gsl cpython llvm libpostal leptonica tesseract caffe cuda mxnet tensorflow tensorrt ale onnx liquidfun skia systems)
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
