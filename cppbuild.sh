#!/bin/bash
# Scripts to build and install native C++ libraries
set -eu

which cmake3 &> /dev/null && CMAKE3="cmake3" || CMAKE3="cmake"
[[ -z ${CMAKE:-} ]] && CMAKE=$CMAKE3
[[ -z ${MAKEJ:-} ]] && MAKEJ=4
[[ -z ${OLDCC:-} ]] && OLDCC="gcc"
[[ -z ${OLDCXX:-} ]] && OLDCXX="g++"
[[ -z ${OLDFC:-} ]] && OLDFC="gfortran"

KERNEL=(`uname -s | tr '[A-Z]' '[a-z]'`)
ARCH=(`uname -m | tr '[A-Z]' '[a-z]'`)
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
    arm64)
        ARCH=arm64
        ;;
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

echo -n "Building platform \"$PLATFORM\""
if [[ -n "$EXTENSION" ]]; then
    echo -n " with extension \"$EXTENSION\""
fi
echo

if [[ -z ${OPERATION:-} ]]; then
    echo "Usage: ANDROID_NDK=/path/to/android-ndk/ bash cppbuild.sh [-platform <name>] [-extension <name>] <install | clean> [projects]"
    echo "where possible platform names are: android-arm, android-x86, linux-x86, linux-x86_64, macosx-arm64, macosx-x86_64, windows-x86, windows-x86_64, etc."
    exit 1
fi

if [[ -z ${ANDROID_NDK:-} ]]; then
    ANDROID_NDK=~/Android/android-ndk/
fi
export ANDROID_NDK
export ANDROID_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/$KERNEL-$ARCH/bin/clang"
export ANDROID_CPP="$ANDROID_NDK/sources/cxx-stl/llvm-libc++/"
case $PLATFORM in
    android-arm)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/arm-linux-androideabi-4.9/prebuilt/$KERNEL-$ARCH/"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-24/arch-arm/"
        export ANDROID_FLAGS="--gcc-toolchain=$ANDROID_BIN --sysroot=$ANDROID_ROOT -isystem $ANDROID_CPP/include/ -isystem $ANDROID_CPP/../llvm-libc++abi/include/ -isystem $ANDROID_NDK/sources/android/support/include/ -isystem $ANDROID_CPP/libs/armeabi-v7a/include/ -isystem $ANDROID_NDK/sysroot/usr/include/ -isystem $ANDROID_NDK/sysroot/usr/include/arm-linux-androideabi/ -isystem $ANDROID_ROOT/usr/include/ -isystem $ANDROID_NDK/sources/android/cpufeatures/ -D__ANDROID_API__=24 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector-strong -target armv7-none-linux-androideabi -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -z text -Wno-unused-command-line-argument -Wno-unknown-warning-option -Wno-ignored-optimization-argument -Wl,--fix-cortex-a8 -L$ANDROID_CPP/libs/armeabi-v7a/ -L$ANDROID_ROOT/usr/lib/ -Wl,--no-undefined -nostdlib++"
        export ANDROID_PREFIX="$ANDROID_BIN/bin/arm-linux-androideabi"
        export ANDROID_LIBS="-llog -lc++_static -lc++abi -landroid_support -ldl -lm -lc"
        ;;
    android-arm64)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/aarch64-linux-android-4.9/prebuilt/$KERNEL-$ARCH/"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-24/arch-arm64/"
        export ANDROID_FLAGS="--gcc-toolchain=$ANDROID_BIN --sysroot=$ANDROID_ROOT -isystem $ANDROID_CPP/include/ -isystem $ANDROID_CPP/../llvm-libc++abi/include/ -isystem $ANDROID_NDK/sources/android/support/include/ -isystem $ANDROID_CPP/libs/arm64-v8a/include/ -isystem $ANDROID_NDK/sysroot/usr/include/ -isystem $ANDROID_NDK/sysroot/usr/include/aarch64-linux-android/ -isystem $ANDROID_ROOT/usr/include/ -isystem $ANDROID_NDK/sources/android/cpufeatures/ -D__ANDROID_API__=24 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector-strong -target aarch64-none-linux-android -march=armv8-a -z text -Wno-unused-command-line-argument -Wno-unknown-warning-option -Wno-ignored-optimization-argument -L$ANDROID_CPP/libs/arm64-v8a/ -L$ANDROID_ROOT/usr/lib/ -Wl,--no-undefined -nostdlib++"
        export ANDROID_PREFIX="$ANDROID_BIN/bin/aarch64-linux-android"
        export ANDROID_LIBS="-llog -lc++_static -lc++abi -ldl -lm -lc"
        ;;
    android-x86)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/x86-4.9/prebuilt/$KERNEL-$ARCH/"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-24/arch-x86/"
        export ANDROID_FLAGS="--gcc-toolchain=$ANDROID_BIN --sysroot=$ANDROID_ROOT -isystem $ANDROID_CPP/include/ -isystem $ANDROID_CPP/../llvm-libc++abi/include/ -isystem $ANDROID_NDK/sources/android/support/include/ -isystem $ANDROID_CPP/libs/x86/include/ -isystem $ANDROID_NDK/sysroot/usr/include/ -isystem $ANDROID_NDK/sysroot/usr/include/i686-linux-android/ -isystem $ANDROID_ROOT/usr/include/ -isystem $ANDROID_NDK/sources/android/cpufeatures/ -D__ANDROID_API__=24 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector-strong -target i686-none-linux-android -march=i686 -mtune=atom -mssse3 -mfpmath=sse -z text -Wno-unused-command-line-argument -Wno-unknown-warning-option -Wno-ignored-optimization-argument -L$ANDROID_CPP/libs/x86/ -L$ANDROID_ROOT/usr/lib/ -Wl,--no-undefined -nostdlib++"
        export ANDROID_PREFIX="$ANDROID_BIN/bin/i686-linux-android"
        export ANDROID_LIBS="-llog -lc++_static -lc++abi -landroid_support -ldl -lm -lc"
        ;;
    android-x86_64)
        export ANDROID_BIN="$ANDROID_NDK/toolchains/x86_64-4.9/prebuilt/$KERNEL-$ARCH/"
        export ANDROID_ROOT="$ANDROID_NDK/platforms/android-24/arch-x86_64/"
        export ANDROID_FLAGS="--gcc-toolchain=$ANDROID_BIN --sysroot=$ANDROID_ROOT -isystem $ANDROID_CPP/include/ -isystem $ANDROID_CPP/../llvm-libc++abi/include/ -isystem $ANDROID_NDK/sources/android/support/include/ -isystem $ANDROID_CPP/libs/x86_64/include/ -isystem $ANDROID_NDK/sysroot/usr/include/ -isystem $ANDROID_NDK/sysroot/usr/include/x86_64-linux-android/ -isystem $ANDROID_ROOT/usr/include/ -isystem $ANDROID_NDK/sources/android/cpufeatures/ -D__ANDROID_API__=24 -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector-strong -target x86_64-none-linux-android -march=x86-64 -mtune=atom -z text -Wno-unused-command-line-argument -Wno-unknown-warning-option -Wno-ignored-optimization-argument -L$ANDROID_CPP/libs/x86_64/ -L$ANDROID_ROOT/usr/lib64/ -Wl,--no-undefined -nostdlib++"
        export ANDROID_PREFIX="$ANDROID_BIN/bin/x86_64-linux-android"
        export ANDROID_LIBS="-llog -lc++_static -lc++abi -ldl -lm -lc"
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
    PROJECTS=(opencv ffmpeg flycapture spinnaker libdc1394 libfreenect libfreenect2 librealsense librealsense2 videoinput artoolkitplus chilitags flandmark arrow hdf5 hyperscan lz4 mkl mkl-dnn dnnl openblas arpack-ng cminpack fftw gsl cpython numpy scipy gym llvm libpostal leptonica tesseract caffe openpose cuda nvcodec opencl mxnet pytorch tensorflow tensorflow-lite tensorrt tritonserver depthai ale onnx ngraph onnxruntime tvm liquidfun qt skia cpu_features modsecurity systems)
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
