// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


//
//  Define the generic mapping array.  This is used to denote the
//  mapping of each generic access right to a specific access mask.
//

@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class GENERIC_MAPPING extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public GENERIC_MAPPING() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public GENERIC_MAPPING(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GENERIC_MAPPING(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public GENERIC_MAPPING position(long position) {
        return (GENERIC_MAPPING)super.position(position);
    }
    @Override public GENERIC_MAPPING getPointer(long i) {
        return new GENERIC_MAPPING((Pointer)this).offsetAddress(i);
    }

    public native @Cast("ACCESS_MASK") int GenericRead(); public native GENERIC_MAPPING GenericRead(int setter);
    public native @Cast("ACCESS_MASK") int GenericWrite(); public native GENERIC_MAPPING GenericWrite(int setter);
    public native @Cast("ACCESS_MASK") int GenericExecute(); public native GENERIC_MAPPING GenericExecute(int setter);
    public native @Cast("ACCESS_MASK") int GenericAll(); public native GENERIC_MAPPING GenericAll(int setter);
}
