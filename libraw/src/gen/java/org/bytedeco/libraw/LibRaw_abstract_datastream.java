// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.libraw;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.libraw.global.LibRaw.*;


@Properties(inherit = org.bytedeco.libraw.presets.LibRaw.class)
public class LibRaw_abstract_datastream extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LibRaw_abstract_datastream(Pointer p) { super(p); }

  public native int valid();
  public native int read(Pointer arg0, @Cast("size_t") long arg1, @Cast("size_t") long arg2);
  public native int seek(@Cast("INT64") long arg0, int arg1);
  public native @Cast("INT64") long tell();
  public native @Cast("INT64") long size();
  public native int get_char();
  public native @Cast("char*") BytePointer gets(@Cast("char*") BytePointer arg0, int arg1);
  public native @Cast("char*") ByteBuffer gets(@Cast("char*") ByteBuffer arg0, int arg1);
  public native @Cast("char*") byte[] gets(@Cast("char*") byte[] arg0, int arg1);
  public native int scanf_one(@Cast("const char*") BytePointer arg0, Pointer arg1);
  public native int scanf_one(String arg0, Pointer arg1);
  public native int eof();
// #ifdef LIBRAW_OLD_VIDEO_SUPPORT
// #endif
  public native int jpeg_src(Pointer arg0);
  public native void buffering_off();
  /* reimplement in subclass to use parallel access in xtrans_load_raw() if
   * OpenMP is not used */
  public native int lock(); /* success */
  public native void unlock();
  public native @Cast("const char*") BytePointer fname();
// #ifdef LIBRAW_WIN32_UNICODEPATHS
// #endif
}
