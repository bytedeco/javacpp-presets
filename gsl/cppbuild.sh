if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

if [[ $PLATFORM == windows* ]]; then
    GSL_VERSION=1.16-2
    download http://mirrors.kernel.org/fedora/development/rawhide/x86_64/os/Packages/m/mingw32-gsl-$GSL_VERSION.fc21.noarch.rpm mingw32-gsl-$GSL_VERSION.rpm
    download http://mirrors.kernel.org/fedora/development/rawhide/x86_64/os/Packages/m/mingw64-gsl-$GSL_VERSION.fc21.noarch.rpm mingw64-gsl-$GSL_VERSION.rpm

    INSTALL_DIR=/C/MinGW/local
    mkdir -p $INSTALL_DIR/include $INSTALL_DIR/lib32 $INSTALL_DIR/lib64 $INSTALL_DIR/bin32 $INSTALL_DIR/bin64
else
    GSL_VERSION=1.16
    download ftp://ftp.gnu.org/gnu/gsl/gsl-$GSL_VERSION.tar.gz gsl-$GSL_VERSION.tar.gz

    tar -xzvf gsl-$GSL_VERSION.tar.gz
    mv gsl-$GSL_VERSION gsl-$GSL_VERSION-$PLATFORM
    cd gsl-$GSL_VERSION-$PLATFORM
fi

case $PLATFORM in
    android-arm)
        ./configure --host="arm-linux-androideabi" --prefix="$ANDROID_NDK/../local/" --libdir="$ANDROID_NDK/../local/lib/armeabi/" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ;;
     android-x86)
        ./configure --host="i686-linux-android" --prefix="$ANDROID_NDK/../local/" --libdir="$ANDROID_NDK/../local/lib/x86/" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ;;
    linux-x86)
        ./configure --prefix=/usr/local/ --libdir=/usr/local/lib32/ CC="gcc -m32"
        make -j4
        sudo make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=/usr/local/ --libdir=/usr/local/lib64/
        make -j4
        sudo make install-strip
        ;;
    macosx-x86_64)
        ./configure
        make -j4
        sudo make install-strip
        ;;
    windows-x86)
        /C/Program\ Files/7-Zip/7z x mingw32-gsl-$GSL_VERSION.rpm
        /C/Program\ Files/7-Zip/7z x mingw32-gsl-$GSL_VERSION.cpio
        cd usr/i686-w64-mingw32/sys-root/mingw
        echo LIBRARY libgsl-0.dll > libgsl-0.def
        echo EXPORTS >> libgsl-0.def
        dumpbin //exports bin/libgsl-0.dll | tail -n +20 | head -n -15 | cut -c27- >> libgsl-0.def
        lib /def:libgsl-0.def /out:libgsl-0.lib /machine:x86
        echo LIBRARY libgslcblas-0.dll > libgslcblas-0.def
        echo EXPORTS >> libgslcblas-0.def
        dumpbin //exports bin/libgslcblas-0.dll | tail -n +20 | head -n -15 | cut -c27- >> libgslcblas-0.def
        lib /def:libgslcblas-0.def /out:libgslcblas-0.lib /machine:x86
        cp -a include/gsl $INSTALL_DIR/include
        cp -a *.lib $INSTALL_DIR/lib32
        cp -a bin/*.dll $INSTALL_DIR/bin32
        cd ../../../../
        ;;
    windows-x86_64)
        /C/Program\ Files/7-Zip/7z x mingw64-gsl-$GSL_VERSION.rpm
        /C/Program\ Files/7-Zip/7z x mingw64-gsl-$GSL_VERSION.cpio
        cd usr/x86_64-w64-mingw32/sys-root/mingw
        echo LIBRARY libgsl-0.dll > libgsl-0.def
        echo EXPORTS >> libgsl-0.def
        dumpbin //exports bin/libgsl-0.dll | tail -n +20 | head -n -15 | cut -c27- >> libgsl-0.def
        lib /def:libgsl-0.def /out:libgsl-0.lib /machine:x64
        echo LIBRARY libgslcblas-0.dll > libgslcblas-0.def
        echo EXPORTS >> libgslcblas-0.def
        dumpbin //exports bin/libgslcblas-0.dll | tail -n +20 | head -n -15 | cut -c27- >> libgslcblas-0.def
        lib /def:libgslcblas-0.def /out:libgslcblas-0.lib /machine:x64
        cp -a include/gsl $INSTALL_DIR/include
        cp -a *.lib $INSTALL_DIR/lib64
        cp -a bin/*.dll $INSTALL_DIR/bin64
        cd ../../../../
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

if [[ $PLATFORM != windows* ]]; then
    cd ..
fi
