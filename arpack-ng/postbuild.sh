#!/bin/bash
# Fix GCC library incorrectly linking for rpath on Mac
for VER in {5..12}; do
    for LIB in libarpack.2.dylib libjniarpack.dylib; do
        LIBFILE=target/native/org/bytedeco/arpackng/macosx-x86_64/$LIB
        if [[ -f $LIBFILE ]] && [[ -f /usr/local/lib/gcc/$VER/libgcc_s.1.dylib ]]; then
            echo Fixing $LIBFILE for GCC $VER
            install_name_tool -change /usr/local/lib/gcc/$VER/libgcc_s.1.dylib @rpath/libgcc_s.1.dylib $LIBFILE
        fi
    done
done
