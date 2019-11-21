#!/bin/bash
# Fix GCC library incorrectly linking for rpath on Mac
for VER in 5 6 7 8 9; do
    for LIB in libopenblas.0.dylib libjniopenblas.dylib; do
        LIBFILE=target/native/org/bytedeco/openblas/macosx-x86_64/$LIB
        if [[ -f $LIBFILE ]]; then
            echo Fixing $LIBFILE
            install_name_tool -change /usr/local/lib/gcc/$VER/libgcc_s.1.dylib @rpath/libgcc_s.1.dylib $LIBFILE
        fi
    done
done
