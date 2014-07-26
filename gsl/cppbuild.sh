if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

if [[ $PLATFORM == windows* ]]; then
    GSL_VERSION=1.16-2
    [[ $PLATFORM == *64 ]] && BITS=64 || BITS=32
    download http://mirrors.kernel.org/fedora/development/rawhide/x86_64/os/Packages/m/mingw$BITS-gsl-$GSL_VERSION.fc21.noarch.rpm mingw$BITS-gsl-$GSL_VERSION.rpm

    mkdir -p $PLATFORM
    cd $PLATFORM
    mkdir -p include lib bin
    /C/Program\ Files/7-Zip/7z x -y ../mingw$BITS-gsl-$GSL_VERSION.rpm -o..
    /C/Program\ Files/7-Zip/7z x -y ../mingw$BITS-gsl-$GSL_VERSION.cpio
else
    GSL_VERSION=1.16
    download ftp://ftp.gnu.org/gnu/gsl/gsl-$GSL_VERSION.tar.gz gsl-$GSL_VERSION.tar.gz

    mkdir -p $PLATFORM
    cd $PLATFORM
    INSTALL_PATH=`pwd`
    tar -xzvf ../gsl-$GSL_VERSION.tar.gz
    cd gsl-$GSL_VERSION
fi

case $PLATFORM in
    android-arm)
        ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ;;
     android-x86)
        ./configure --prefix=$INSTALL_PATH --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ;;
    linux-x86)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m32"
        make -j4
        make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m64"
        make -j4
        make install-strip
        ;;
    macosx-*)
        ./configure --prefix=$INSTALL_PATH
        make -j4
        make install-strip
        ;;
    windows-x86)
        cd usr/i686-w64-mingw32/sys-root/mingw
        echo LIBRARY libgsl-0.dll > libgsl-0.def
        echo EXPORTS >> libgsl-0.def
        dumpbin //exports bin/libgsl-0.dll | tail -n +20 | head -n -15 | cut -c27- >> libgsl-0.def
        lib /def:libgsl-0.def /out:libgsl-0.lib /machine:x86
        echo LIBRARY libgslcblas-0.dll > libgslcblas-0.def
        echo EXPORTS >> libgslcblas-0.def
        dumpbin //exports bin/libgslcblas-0.dll | tail -n +20 | head -n -15 | cut -c27- >> libgslcblas-0.def
        lib /def:libgslcblas-0.def /out:libgslcblas-0.lib /machine:x86
        cp -r include/* ../../../../include
        cp *.lib ../../../../lib
        cp -r bin/* ../../../../bin
        cd ../../../..
        ;;
    windows-x86_64)
        cd usr/x86_64-w64-mingw32/sys-root/mingw
        echo LIBRARY libgsl-0.dll > libgsl-0.def
        echo EXPORTS >> libgsl-0.def
        dumpbin //exports bin/libgsl-0.dll | tail -n +20 | head -n -15 | cut -c27- >> libgsl-0.def
        lib /def:libgsl-0.def /out:libgsl-0.lib /machine:x64
        echo LIBRARY libgslcblas-0.dll > libgslcblas-0.def
        echo EXPORTS >> libgslcblas-0.def
        dumpbin //exports bin/libgslcblas-0.dll | tail -n +20 | head -n -15 | cut -c27- >> libgslcblas-0.def
        lib /def:libgslcblas-0.def /out:libgslcblas-0.lib /machine:x64
        cp -r include/* ../../../../include
        cp *.lib ../../../../lib
        cp -r bin/* ../../../../bin
        cd ../../../..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
