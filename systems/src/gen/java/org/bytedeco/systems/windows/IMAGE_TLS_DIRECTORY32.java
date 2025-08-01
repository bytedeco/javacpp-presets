// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class IMAGE_TLS_DIRECTORY32 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public IMAGE_TLS_DIRECTORY32() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public IMAGE_TLS_DIRECTORY32(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IMAGE_TLS_DIRECTORY32(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public IMAGE_TLS_DIRECTORY32 position(long position) {
        return (IMAGE_TLS_DIRECTORY32)super.position(position);
    }
    @Override public IMAGE_TLS_DIRECTORY32 getPointer(long i) {
        return new IMAGE_TLS_DIRECTORY32((Pointer)this).offsetAddress(i);
    }

    public native @Cast("DWORD") int StartAddressOfRawData(); public native IMAGE_TLS_DIRECTORY32 StartAddressOfRawData(int setter);
    public native @Cast("DWORD") int EndAddressOfRawData(); public native IMAGE_TLS_DIRECTORY32 EndAddressOfRawData(int setter);
    public native @Cast("DWORD") int AddressOfIndex(); public native IMAGE_TLS_DIRECTORY32 AddressOfIndex(int setter);             // PDWORD
    public native @Cast("DWORD") int AddressOfCallBacks(); public native IMAGE_TLS_DIRECTORY32 AddressOfCallBacks(int setter);         // PIMAGE_TLS_CALLBACK *
    public native @Cast("DWORD") int SizeOfZeroFill(); public native IMAGE_TLS_DIRECTORY32 SizeOfZeroFill(int setter);
        public native @Cast("DWORD") int Characteristics(); public native IMAGE_TLS_DIRECTORY32 Characteristics(int setter);
            public native @Cast("DWORD") @NoOffset int Reserved0(); public native IMAGE_TLS_DIRECTORY32 Reserved0(int setter);
            public native @Cast("DWORD") @NoOffset int Alignment(); public native IMAGE_TLS_DIRECTORY32 Alignment(int setter);
            public native @Cast("DWORD") @NoOffset int Reserved1(); public native IMAGE_TLS_DIRECTORY32 Reserved1(int setter);  

}
