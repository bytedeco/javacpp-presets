# CMake toolchain to build flandmark for Android 2.2. Sample usage:
#
# ANDROID_BIN=`pwd`/../android-ndk/toolchains/x86-4.6/prebuilt/linux-x86_64/bin/i686-linux-android \
# ANDROID_CPP=`pwd`/../android-ndk/sources/cxx-stl/gnu-libstdc++/4.6/ \
# ANDROID_ROOT=`pwd`/../android-ndk/platforms/android-9/arch-x86/ \
# cmake -DCMAKE_TOOLCHAIN_FILE=android-x86.cmake -DCMAKE_INSTALL_PREFIX=..

set(CMAKE_SYSTEM_NAME UnixPaths)
set(CMAKE_SYSTEM_PROCESSOR atom)
set(ANDROID TRUE)

set(CMAKE_C_COMPILER   "$ENV{ANDROID_BIN}-gcc")
set(CMAKE_CXX_COMPILER "$ENV{ANDROID_BIN}-g++")

set(CMAKE_C_LINK_EXECUTABLE    "<CMAKE_C_COMPILER>   <FLAGS> <CMAKE_C_LINK_FLAGS>   <LINK_FLAGS> <OBJECTS> -o <TARGET> -nostdlib <LINK_LIBRARIES> -L$ENV{ANDROID_ROOT}/usr/lib/ -lgcc -ldl -lz -lm -lc")
set(CMAKE_CXX_LINK_EXECUTABLE  "<CMAKE_CXX_COMPILER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> -nostdlib <LINK_LIBRARIES> -L$ENV{ANDROID_CPP}/libs/x86/ -L$ENV{ANDROID_ROOT}/usr/lib/ -lgnustl_static -lgcc -ldl -lz -lm -lc")

set(CMAKE_C_CREATE_SHARED_LIBRARY    "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> <CMAKE_SHARED_LIBRARY_SONAME_C_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> -nostdlib <LINK_LIBRARIES> -L$ENV{ANDROID_ROOT}/usr/lib/ -lgcc -ldl -lz -lm -lc")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY  "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> -nostdlib <LINK_LIBRARIES> -L$ENV{ANDROID_CPP}/libs/x86/ -L$ENV{ANDROID_ROOT}/usr/lib/ -lgnustl_static -lgcc -ldl -lz -lm -lc")

add_definitions("-DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -mtune=atom -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300")

include_directories("$ENV{ANDROID_CPP}/include/" "$ENV{ANDROID_CPP}/libs/x86/include/" "$ENV{ANDROID_ROOT}/usr/include/")
