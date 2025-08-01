// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


// #ifdef UNICODE
// #else
// #endif // !UNICODE

@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class WIN32_FILE_ATTRIBUTE_DATA extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public WIN32_FILE_ATTRIBUTE_DATA() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public WIN32_FILE_ATTRIBUTE_DATA(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public WIN32_FILE_ATTRIBUTE_DATA(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public WIN32_FILE_ATTRIBUTE_DATA position(long position) {
        return (WIN32_FILE_ATTRIBUTE_DATA)super.position(position);
    }
    @Override public WIN32_FILE_ATTRIBUTE_DATA getPointer(long i) {
        return new WIN32_FILE_ATTRIBUTE_DATA((Pointer)this).offsetAddress(i);
    }

    public native @Cast("DWORD") int dwFileAttributes(); public native WIN32_FILE_ATTRIBUTE_DATA dwFileAttributes(int setter);
    public native @ByRef FILETIME ftCreationTime(); public native WIN32_FILE_ATTRIBUTE_DATA ftCreationTime(FILETIME setter);
    public native @ByRef FILETIME ftLastAccessTime(); public native WIN32_FILE_ATTRIBUTE_DATA ftLastAccessTime(FILETIME setter);
    public native @ByRef FILETIME ftLastWriteTime(); public native WIN32_FILE_ATTRIBUTE_DATA ftLastWriteTime(FILETIME setter);
    public native @Cast("DWORD") int nFileSizeHigh(); public native WIN32_FILE_ATTRIBUTE_DATA nFileSizeHigh(int setter);
    public native @Cast("DWORD") int nFileSizeLow(); public native WIN32_FILE_ATTRIBUTE_DATA nFileSizeLow(int setter);
}
