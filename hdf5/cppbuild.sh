#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" hdf5
    popd
    exit
fi

ZLIB=zlib-1.2.13
HDF5_VERSION=1.14.0
AEC_VERSION=1.0.6
download "http://zlib.net/$ZLIB.tar.gz" $ZLIB.tar.gz
download "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.bz2" hdf5-$HDF5_VERSION.tar.bz2
# Use Github mirror repo rather than Gitlab repo for download speed
#download "https://gitlab.dkrz.de/k202009/libaec/uploads/45b10e42123edd26ab7b3ad92bcf7be2/libaec-$AEC_VERSION.tar.gz" libaec-$AEC_VERSION.tar.gz
download "https://github.com/MathisRosenhauer/libaec/releases/download/v$AEC_VERSION/libaec-$AEC_VERSION.tar.gz" libaec-$AEC_VERSION.tar.gz

mkdir -p $PLATFORM
pushd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xf ../hdf5-$HDF5_VERSION.tar.bz2
tar --totals -xf ../libaec-$AEC_VERSION.tar.gz
pushd hdf5-$HDF5_VERSION

#sedinplace '/cmake_minimum_required/d' $(find ./ -iname CMakeLists.txt)
sedinplace 's/# *cmakedefine/#cmakedefine/g' config/cmake/H5pubconf.h.in
sedinplace 's/COMPATIBILITY SameMinorVersion/COMPATIBILITY AnyNewerVersion/g' CMakeInstallation.cmake
sedinplace '/C_RUN (/{N;N;d;}' config/cmake/ConfigureChecks.cmake

# As of 1.14.0 the integrated cmake process for building aec/szip is broken
# Revisit integrated szip build with 1.14.1

case $PLATFORM in
# HDF5 does not currently support cross-compiling:
# https://support.hdfgroup.org/HDF5/faq/compile.html
#    android-arm)
#        # Build libaec for szip first
#        mkdir -p ../libaec-$AEC_VERSION/build
#        pushd ../libaec-$AEC_VERSION/build
#        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
#        make -j $MAKEJ
#        make install
#        popd
#
#        patch -Np1 < ../../../hdf5-android.patch
#        ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" AR="$ANDROID_BIN-ar" RANLIB="$ANDROID_BIN-ranlib" CPP="$ANDROID_BIN-cpp" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/armeabi/include/ -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/armeabi/ -nostdlib -Wl,--fix-cortex-a8 -z text -L./" LIBS="-lgnustl_static -lgcc -ldl -lz -lm -lc" --enable-cxx --enable-java
#        make -j $MAKEJ
#        make install-strip
#        ;;
#    android-x86)
#        # Build libaec for szip first
#        mkdir -p ../libaec-$AEC_VERSION/build
#        pushd ../libaec-$AEC_VERSION/build
#        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
#        make -j $MAKEJ
#        make install
#        popd
#
#        patch -Np1 < ../../../hdf5-android.patch
#        ./configure --prefix=$INSTALL_PATH --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" AR="$ANDROID_BIN-ar" RANLIB="$ANDROID_BIN-ranlib" CPP="$ANDROID_BIN-cpp" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/x86/include/ -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/x86/ -nostdlib -z text -L." LIBS="-lgnustl_static -lgcc -ldl -lz -lm -lc" --enable-cxx --enable-java
#        make -j $MAKEJ
#        make install-strip
#        ;;
    linux-armhf)
        # Build libaec for szip first
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ arm ]]; then
          ./configure --prefix=$INSTALL_PATH CC="gcc" CXX="g++" --enable-cxx --enable-java
          make -j $MAKEJ
          make install-strip
        else
          echo "Not native arm so assume cross compiling"
          patch -Np1 < ../../../hdf5-linux-armhf.patch || true
          #need this to run twice, first run fails so we fake the exit code too
          for x in 1 2; do
              "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=`pwd`/arm.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE" -DCMAKE_C_FLAGS="-D_GNU_SOURCE" -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_JAVA=ON . || true
          done
          make -j $MAKEJ
          make install
        fi
        ;;
    linux-arm64)
        # Build libaec for szip first
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ arm ]]; then
          ./configure --prefix=$INSTALL_PATH CC="gcc -m64" CXX="g++ -m64" --enable-cxx --enable-java
          make -j $MAKEJ
          make install-strip
        else
          echo "Not native arm so assume cross compiling"
          patch -Np1 < ../../../hdf5-linux-arm64.patch || true
          #need this to run twice, first run fails so we fake the exit code too
          for x in 1 2; do
              "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=`pwd`/arm64.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE" -DCMAKE_C_FLAGS="-D_GNU_SOURCE" -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_JAVA=ON . || true
          done
          make -j $MAKEJ
          make install
        fi
        ;;
    linux-x86)
        # Build libaec for szip first
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        ./configure --prefix=$INSTALL_PATH CC="gcc -m32" CXX="g++ -m32" --enable-cxx --enable-java --with-szlib
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        # Build libaec for szip first
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        ./configure --prefix=$INSTALL_PATH CC="gcc -m64" CXX="g++ -m64" --enable-cxx --enable-java --with-szlib
        make -j $MAKEJ
        make install-strip
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          # Build libaec for szip first
          mkdir -p ../libaec-$AEC_VERSION/build
          pushd ../libaec-$AEC_VERSION/build
          "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
          make -j $MAKEJ
          make install
          popd

          ./configure --prefix=$INSTALL_PATH CC="gcc -m64" CXX="g++ -m64" --enable-cxx --enable-java --with-szlib
          make -j $MAKEJ
          make install-strip
        else
          echo "Not native ppc so assume cross compiling"
          patch -Np1 < ../../../hdf5-linux-ppc64le.patch || true
          #need this to run twice, first run fails so we fake the exit code too
          for x in 1 2; do
              "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=`pwd`/ppc.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE" -DCMAKE_C_FLAGS="-D_GNU_SOURCE" -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DSZAEC_TGZ_NAME:STRING="libaec-$AEC_VERSION.tar.gz" -DHDF5_ENABLE_SZIP_SUPPORT=ON -DHDF5_ENABLE_SZIP_ENCODING=ON -DUSE_LIBAEC=ON -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_JAVA=ON . || true
          done
          make -j $MAKEJ
          make install
        fi
        ;;
    macosx-*)
        # Build libaec for szip first
        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        make -j $MAKEJ
        make install
        popd

        patch -Np1 < ../../../hdf5-macosx.patch
        ./configure --prefix=$INSTALL_PATH --enable-cxx --enable-java --with-szlib
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        export CC="cl.exe"
        export CXX="cl.exe"

        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        ninja -j $MAKEJ
        ninja install
        popd

        mkdir -p build
        pushd build
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=ON -DHDF5_ENABLE_SZIP_ENCODING=ON -DUSE_LIBAEC=ON -DSZIP_LIBRARY:FILEPATH="$INSTALL_PATH/lib/szip_static.lib" -DSZIP_INCLUDE_DIR="$INSTALL_PATH/include" -DSZIP_USE_EXTERNAL:BOOL=OFF -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_JAVA=ON ..
        sedinplace 's/Release\\libz.lib/zlibstatic.lib/g' build.ninja
        ninja -j $MAKEJ HDF5_ZLIB
        ninja -j $MAKEJ
        ninja install
        cp bin/zlib* ../../lib/
        popd
        ;;
    windows-x86_64)
        export CC="cl.exe"
        export CXX="cl.exe"

        mkdir -p ../libaec-$AEC_VERSION/build
        pushd ../libaec-$AEC_VERSION/build
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ..
        ninja -j $MAKEJ
        ninja install
        popd

        mkdir -p build
        pushd build
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=ON -DHDF5_ENABLE_SZIP_ENCODING=ON -DUSE_LIBAEC=ON -DSZIP_LIBRARY:FILEPATH="$INSTALL_PATH/lib/szip_static.lib" -DSZIP_INCLUDE_DIR="$INSTALL_PATH/include" -DSZIP_USE_EXTERNAL:BOOL=OFF -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_JAVA=ON ..

        sedinplace 's/Release\\libz.lib/zlibstatic.lib/g' build.ninja
        ninja -j $MAKEJ HDF5_ZLIB
        ninja -j $MAKEJ
        ninja install
        cp bin/zlib* ../../lib/
        popd
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

[ -d "../java" ] && rm -r ../java
cp -r java/src ../java

# Return to cppbuild directory
popd
