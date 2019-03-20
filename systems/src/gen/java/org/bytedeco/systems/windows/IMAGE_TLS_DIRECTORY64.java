// Targeted by JavaCPP version 1.5-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class IMAGE_TLS_DIRECTORY64 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public IMAGE_TLS_DIRECTORY64() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public IMAGE_TLS_DIRECTORY64(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IMAGE_TLS_DIRECTORY64(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public IMAGE_TLS_DIRECTORY64 position(long position) {
        return (IMAGE_TLS_DIRECTORY64)super.position(position);
    }

    public native @Cast("ULONGLONG") long StartAddressOfRawData(); public native IMAGE_TLS_DIRECTORY64 StartAddressOfRawData(long StartAddressOfRawData);
    public native @Cast("ULONGLONG") long EndAddressOfRawData(); public native IMAGE_TLS_DIRECTORY64 EndAddressOfRawData(long EndAddressOfRawData);
    public native @Cast("ULONGLONG") long AddressOfIndex(); public native IMAGE_TLS_DIRECTORY64 AddressOfIndex(long AddressOfIndex);         // PDWORD
    public native @Cast("ULONGLONG") long AddressOfCallBacks(); public native IMAGE_TLS_DIRECTORY64 AddressOfCallBacks(long AddressOfCallBacks);     // PIMAGE_TLS_CALLBACK *;
    public native @Cast("DWORD") int SizeOfZeroFill(); public native IMAGE_TLS_DIRECTORY64 SizeOfZeroFill(int SizeOfZeroFill);
        public native @Cast("DWORD") int Characteristics(); public native IMAGE_TLS_DIRECTORY64 Characteristics(int Characteristics);
            public native @Cast("DWORD") @NoOffset int Reserved0(); public native IMAGE_TLS_DIRECTORY64 Reserved0(int Reserved0);
            public native @Cast("DWORD") @NoOffset int Alignment(); public native IMAGE_TLS_DIRECTORY64 Alignment(int Alignment);
            public native @Cast("DWORD") @NoOffset int Reserved1(); public native IMAGE_TLS_DIRECTORY64 Reserved1(int Reserved1);  

}