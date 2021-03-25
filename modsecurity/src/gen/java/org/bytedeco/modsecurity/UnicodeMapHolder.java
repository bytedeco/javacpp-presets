// Targeted by JavaCPP version 1.5.6-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.modsecurity;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.modsecurity.global.modsecurity.*;



@Namespace("modsecurity") @NoOffset @Properties(inherit = org.bytedeco.modsecurity.presets.modsecurity.class)
public class UnicodeMapHolder extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public UnicodeMapHolder(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public UnicodeMapHolder(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public UnicodeMapHolder position(long position) {
        return (UnicodeMapHolder)super.position(position);
    }
    @Override public UnicodeMapHolder getPointer(long i) {
        return new UnicodeMapHolder((Pointer)this).position(position + i);
    }

    public UnicodeMapHolder() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native @ByRef @Name("operator []") IntPointer get(int index);

    public native int at(int index);
    public native void change(int i, int a);

    public native int m_data(int i); public native UnicodeMapHolder m_data(int i, int setter);
    @MemberGetter public native IntPointer m_data();
}