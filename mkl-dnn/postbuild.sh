#!/bin/bash
# Fix GCC library incorrectly linking for rpath on Mac
for VER in 5 6 7 8; do
    for LIB in libmkldnn.0.dylib libjnimkldnn.dylib libjnimklml.dylib; do
        LIBFILE=target/classes/org/bytedeco/javacpp/macosx-x86_64/$LIB
        if [[ -f $LIBFILE ]]; then
            echo Fixing $LIBFILE
            install_name_tool -change /usr/local/lib/gcc/$VER/libgcc_s.1.dylib @rpath/libgcc_s.1.dylib $LIBFILE
        fi
    done
done
