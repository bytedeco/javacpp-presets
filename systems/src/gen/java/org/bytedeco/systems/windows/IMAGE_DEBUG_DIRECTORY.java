// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


//
// Debug Format
//

@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class IMAGE_DEBUG_DIRECTORY extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public IMAGE_DEBUG_DIRECTORY() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public IMAGE_DEBUG_DIRECTORY(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IMAGE_DEBUG_DIRECTORY(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public IMAGE_DEBUG_DIRECTORY position(long position) {
        return (IMAGE_DEBUG_DIRECTORY)super.position(position);
    }
    @Override public IMAGE_DEBUG_DIRECTORY getPointer(long i) {
        return new IMAGE_DEBUG_DIRECTORY((Pointer)this).offsetAddress(i);
    }

    public native @Cast("DWORD") int Characteristics(); public native IMAGE_DEBUG_DIRECTORY Characteristics(int setter);
    public native @Cast("DWORD") int TimeDateStamp(); public native IMAGE_DEBUG_DIRECTORY TimeDateStamp(int setter);
    public native @Cast("WORD") short MajorVersion(); public native IMAGE_DEBUG_DIRECTORY MajorVersion(short setter);
    public native @Cast("WORD") short MinorVersion(); public native IMAGE_DEBUG_DIRECTORY MinorVersion(short setter);
    public native @Cast("DWORD") int Type(); public native IMAGE_DEBUG_DIRECTORY Type(int setter);
    public native @Cast("DWORD") int SizeOfData(); public native IMAGE_DEBUG_DIRECTORY SizeOfData(int setter);
    public native @Cast("DWORD") int AddressOfRawData(); public native IMAGE_DEBUG_DIRECTORY AddressOfRawData(int setter);
    public native @Cast("DWORD") int PointerToRawData(); public native IMAGE_DEBUG_DIRECTORY PointerToRawData(int setter);
}
