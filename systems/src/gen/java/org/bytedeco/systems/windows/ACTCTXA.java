// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class ACTCTXA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ACTCTXA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public ACTCTXA(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ACTCTXA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public ACTCTXA position(long position) {
        return (ACTCTXA)super.position(position);
    }
    @Override public ACTCTXA getPointer(long i) {
        return new ACTCTXA((Pointer)this).offsetAddress(i);
    }

    public native @Cast("ULONG") long cbSize(); public native ACTCTXA cbSize(long setter);
    public native @Cast("DWORD") int dwFlags(); public native ACTCTXA dwFlags(int setter);
    public native @Cast("LPCSTR") BytePointer lpSource(); public native ACTCTXA lpSource(BytePointer setter);
    public native @Cast("USHORT") short wProcessorArchitecture(); public native ACTCTXA wProcessorArchitecture(short setter);
    public native @Cast("LANGID") short wLangId(); public native ACTCTXA wLangId(short setter);
    public native @Cast("LPCSTR") BytePointer lpAssemblyDirectory(); public native ACTCTXA lpAssemblyDirectory(BytePointer setter);
    public native @Cast("LPCSTR") BytePointer lpResourceName(); public native ACTCTXA lpResourceName(BytePointer setter);
    public native @Cast("LPCSTR") BytePointer lpApplicationName(); public native ACTCTXA lpApplicationName(BytePointer setter);
    public native @Cast("HMODULE") Pointer hModule(); public native ACTCTXA hModule(Pointer setter);
}
