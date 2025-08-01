// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class TOKEN_BNO_ISOLATION_INFORMATION extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public TOKEN_BNO_ISOLATION_INFORMATION() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public TOKEN_BNO_ISOLATION_INFORMATION(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TOKEN_BNO_ISOLATION_INFORMATION(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public TOKEN_BNO_ISOLATION_INFORMATION position(long position) {
        return (TOKEN_BNO_ISOLATION_INFORMATION)super.position(position);
    }
    @Override public TOKEN_BNO_ISOLATION_INFORMATION getPointer(long i) {
        return new TOKEN_BNO_ISOLATION_INFORMATION((Pointer)this).offsetAddress(i);
    }

    public native @Cast("PWSTR") CharPointer IsolationPrefix(); public native TOKEN_BNO_ISOLATION_INFORMATION IsolationPrefix(CharPointer setter);
    public native @Cast("BOOLEAN") boolean IsolationEnabled(); public native TOKEN_BNO_ISOLATION_INFORMATION IsolationEnabled(boolean setter);
}
