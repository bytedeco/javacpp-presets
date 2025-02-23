// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.libraw;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.libraw.global.LibRaw.*;


  @Properties(inherit = org.bytedeco.libraw.presets.LibRaw.class)
public class libraw_p1_makernotes_t extends Pointer {
      static { Loader.load(); }
      /** Default native constructor. */
      public libraw_p1_makernotes_t() { super((Pointer)null); allocate(); }
      /** Native array allocator. Access with {@link Pointer#position(long)}. */
      public libraw_p1_makernotes_t(long size) { super((Pointer)null); allocateArray(size); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public libraw_p1_makernotes_t(Pointer p) { super(p); }
      private native void allocate();
      private native void allocateArray(long size);
      @Override public libraw_p1_makernotes_t position(long position) {
          return (libraw_p1_makernotes_t)super.position(position);
      }
      @Override public libraw_p1_makernotes_t getPointer(long i) {
          return new libraw_p1_makernotes_t((Pointer)this).offsetAddress(i);
      }
  
    public native @Cast("char") byte Software(int i); public native libraw_p1_makernotes_t Software(int i, byte setter);
    @MemberGetter public native @Cast("char*") BytePointer Software();        // tag 0x0203
    public native @Cast("char") byte SystemType(int i); public native libraw_p1_makernotes_t SystemType(int i, byte setter);
    @MemberGetter public native @Cast("char*") BytePointer SystemType();      // tag 0x0204
    public native @Cast("char") byte FirmwareString(int i); public native libraw_p1_makernotes_t FirmwareString(int i, byte setter);
    @MemberGetter public native @Cast("char*") BytePointer FirmwareString(); // tag 0x0301
    public native @Cast("char") byte SystemModel(int i); public native libraw_p1_makernotes_t SystemModel(int i, byte setter);
    @MemberGetter public native @Cast("char*") BytePointer SystemModel();
  }
