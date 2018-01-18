#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libuvc2
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    WINDOWSDOWNLAOAD="not clear how to do the windows"
else
    download https://github.com/ktossell/libuvc/archive/master.zip libuvc-master.zip
    mkdir -p $PLATFORM
    cd $PLATFORM
    unzip ../libuvc-master.zip
    cd libuvc-master
fi

case $PLATFORM in
    android-arm)
        CC="$ANDROID_BIN-gcc" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc" ./configure --prefix=.. --static
        make -j4
        make install
        ;;
    android-x86)
        CC="$ANDROID_BIN-gcc" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc" ./configure --prefix=.. --static
        make -j4
        make install
        ;;
    linux-x86)
        CC="gcc -m32 -fPIC" ./configure --prefix=.. --static
        make -j4
        make install
        ;;
    linux-x86_64)
        CC="gcc -m64 -fPIC" ./configure --prefix=.. --static
        make -j4
        make install
        ;;
    macosx-x86_64)
        $CMAKE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
##        ./configure --prefix=.. --static
#   		mkdir build
#   		cd build
#		cmake ..
        make -j4
        make install
        ;;
    windows-x86)
        cp -r include ..
        cp -r lib ..
        mkdir -p ../bin
        cp *.dll ../bin
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..