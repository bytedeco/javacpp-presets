#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" hdf5
    popd
    exit
fi

ZLIB=zlib-1.2.11
HDF5_VERSION=1.12.1
download "http://zlib.net/$ZLIB.tar.gz" $ZLIB.tar.gz
download "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.bz2" hdf5-$HDF5_VERSION.tar.bz2

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xf ../hdf5-$HDF5_VERSION.tar.bz2
cd hdf5-$HDF5_VERSION

sedinplace '/cmake_minimum_required/d' $(find ./ -iname CMakeLists.txt)
sedinplace 's/# *cmakedefine/#cmakedefine/g' config/cmake/H5pubconf.h.in
sedinplace 's/COMPATIBILITY SameMinorVersion/COMPATIBILITY AnyNewerVersion/g' CMakeInstallation.cmake
sedinplace '/C_RUN (/{N;N;d;}' config/cmake/ConfigureChecks.cmake

case $PLATFORM in
# HDF5 does not currently support cross-compiling:
# https://support.hdfgroup.org/HDF5/faq/compile.html
#    android-arm)
#        patch -Np1 < ../../../hdf5-android.patch
#        ./configure --prefix=$INSTALL_PATH --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" AR="$ANDROID_BIN-ar" RANLIB="$ANDROID_BIN-ranlib" CPP="$ANDROID_BIN-cpp" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/armeabi/include/ -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/armeabi/ -nostdlib -Wl,--fix-cortex-a8 -z text -L./" LIBS="-lgnustl_static -lgcc -ldl -lz -lm -lc" --enable-cxx
#        make -j $MAKEJ
#        make install-strip
#        ;;
#    android-x86)
#        patch -Np1 < ../../../hdf5-android.patch
#        ./configure --prefix=$INSTALL_PATH --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" AR="$ANDROID_BIN-ar" RANLIB="$ANDROID_BIN-ranlib" CPP="$ANDROID_BIN-cpp" CC="$ANDROID_BIN-gcc" CXX="$ANDROID_BIN-g++" STRIP="$ANDROID_BIN-strip" CPPFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -I$ANDROID_CPP/include/ -I$ANDROID_CPP/include/backward/ -I$ANDROID_CPP/libs/x86/include/ -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-L$ANDROID_ROOT/usr/lib/ -L$ANDROID_CPP/libs/x86/ -nostdlib -z text -L." LIBS="-lgnustl_static -lgcc -ldl -lz -lm -lc" --enable-cxx
#        make -j $MAKEJ
#        make install-strip
#        ;;
    linux-armhf)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ arm ]]; then
          ./configure --prefix=$INSTALL_PATH CC="gcc" CXX="g++" --enable-cxx
          make -j $MAKEJ
          make install-strip
        else
          echo "Not native arm so assume cross compiling"
          patch -Np1 < ../../../hdf5-linux-armhf.patch || true
          #need this to run twice, first run fails so we fake the exit code too
          for x in 1 2; do
              "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=`pwd`/arm.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE" -DCMAKE_C_FLAGS="-D_GNU_SOURCE" -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_BUILD_CPP_LIB=ON . || true
          done
          make -j $MAKEJ
          make install
        fi
        ;;
    linux-arm64)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ arm ]]; then
          ./configure --prefix=$INSTALL_PATH CC="gcc -m64" CXX="g++ -m64" --enable-cxx
          make -j $MAKEJ
          make install-strip
        else
          echo "Not native arm so assume cross compiling"
          patch -Np1 < ../../../hdf5-linux-arm64.patch || true
          #need this to run twice, first run fails so we fake the exit code too
          for x in 1 2; do
              "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=`pwd`/arm64.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE" -DCMAKE_C_FLAGS="-D_GNU_SOURCE" -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_BUILD_CPP_LIB=ON . || true
          done
          make -j $MAKEJ
          make install
        fi
        ;;
    linux-x86)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m32" CXX="g++ -m32" --enable-cxx
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH CC="gcc -m64" CXX="g++ -m64" --enable-cxx
        make -j $MAKEJ
        make install-strip
        ;;
    linux-ppc64le)
        MACHINE_TYPE=$( uname -m )
        if [[ "$MACHINE_TYPE" =~ ppc64 ]]; then
          ./configure --prefix=$INSTALL_PATH CC="gcc -m64" CXX="g++ -m64" --enable-cxx
          make -j $MAKEJ
          make install-strip
        else
          echo "Not native ppc so assume cross compiling"
          patch -Np1 < ../../../hdf5-linux-ppc64le.patch || true
          #need this to run twice, first run fails so we fake the exit code too
          for x in 1 2; do
              "$CMAKE" -DCMAKE_TOOLCHAIN_FILE=`pwd`/ppc.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE" -DCMAKE_C_FLAGS="-D_GNU_SOURCE" -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_BUILD_CPP_LIB=ON . || true
          done
          make -j $MAKEJ
          make install
        fi
        ;;
    macosx-*)
        patch -Np1 < ../../../hdf5-macosx.patch
        ./configure --prefix=$INSTALL_PATH --enable-cxx
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        mkdir -p build
        cd build
        export CC="cl.exe"
        export CXX="cl.exe"
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_BUILD_CPP_LIB=ON ..
        sedinplace 's/Release\\libzlib.lib/zlibstatic.lib/g' build.ninja
        ninja -j $MAKEJ ZLIB
        ninja -j $MAKEJ
        ninja install
        cp bin/zlib* ../../lib/
        cd ..
        ;;
    windows-x86_64)
        mkdir -p build
        cd build
        export CC="cl.exe"
        export CXX="cl.exe"
        "$CMAKE" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DBUILD_TESTING=false -DHDF5_BUILD_EXAMPLES=false -DHDF5_BUILD_TOOLS=false -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME:STRING="$ZLIB.tar.gz" -DTGZPATH:STRING="$INSTALL_PATH/.." -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_BUILD_CPP_LIB=ON ..
        sedinplace 's/Release\\libzlib.lib/zlibstatic.lib/g' build.ninja
        ninja -j $MAKEJ ZLIB
        ninja -j $MAKEJ
        ninja install
        cp bin/zlib* ../../lib/
        cd ..
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..
