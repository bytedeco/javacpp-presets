// Targeted by JavaCPP version 1.5.8-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libraw;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.libraw.global.LibRaw.*;


  @Properties(inherit = org.bytedeco.libraw.presets.LibRaw.class)
public class libraw_custom_camera_t extends Pointer {
      static { Loader.load(); }
      /** Default native constructor. */
      public libraw_custom_camera_t() { super((Pointer)null); allocate(); }
      /** Native array allocator. Access with {@link Pointer#position(long)}. */
      public libraw_custom_camera_t(long size) { super((Pointer)null); allocateArray(size); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public libraw_custom_camera_t(Pointer p) { super(p); }
      private native void allocate();
      private native void allocateArray(long size);
      @Override public libraw_custom_camera_t position(long position) {
          return (libraw_custom_camera_t)super.position(position);
      }
      @Override public libraw_custom_camera_t getPointer(long i) {
          return new libraw_custom_camera_t((Pointer)this).offsetAddress(i);
      }
  
    public native @Cast("unsigned") int fsize(); public native libraw_custom_camera_t fsize(int setter);
    public native @Cast("ushort") short rw(); public native libraw_custom_camera_t rw(short setter);
    public native @Cast("ushort") short rh(); public native libraw_custom_camera_t rh(short setter);
    public native @Cast("uchar") byte lm(); public native libraw_custom_camera_t lm(byte setter);
    public native @Cast("uchar") byte tm(); public native libraw_custom_camera_t tm(byte setter);
    public native @Cast("uchar") byte rm(); public native libraw_custom_camera_t rm(byte setter);
    public native @Cast("uchar") byte bm(); public native libraw_custom_camera_t bm(byte setter);
    public native @Cast("ushort") short lf(); public native libraw_custom_camera_t lf(short setter);
    public native @Cast("uchar") byte cf(); public native libraw_custom_camera_t cf(byte setter);
    public native @Cast("uchar") byte max(); public native libraw_custom_camera_t max(byte setter);
    public native @Cast("uchar") byte flags(); public native libraw_custom_camera_t flags(byte setter);
    public native @Cast("char") byte t_make(int i); public native libraw_custom_camera_t t_make(int i, byte setter);
    @MemberGetter public native @Cast("char*") BytePointer t_make();
    public native @Cast("char") byte t_model(int i); public native libraw_custom_camera_t t_model(int i, byte setter);
    @MemberGetter public native @Cast("char*") BytePointer t_model();
    public native @Cast("ushort") short offset(); public native libraw_custom_camera_t offset(short setter);
  }