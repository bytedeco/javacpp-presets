// Targeted by JavaCPP version 1.5.8-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libraw;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.libraw.global.LibRaw.*;


  @Properties(inherit = org.bytedeco.libraw.presets.LibRaw.class)
public class libraw_kodak_makernotes_t extends Pointer {
      static { Loader.load(); }
      /** Default native constructor. */
      public libraw_kodak_makernotes_t() { super((Pointer)null); allocate(); }
      /** Native array allocator. Access with {@link Pointer#position(long)}. */
      public libraw_kodak_makernotes_t(long size) { super((Pointer)null); allocateArray(size); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public libraw_kodak_makernotes_t(Pointer p) { super(p); }
      private native void allocate();
      private native void allocateArray(long size);
      @Override public libraw_kodak_makernotes_t position(long position) {
          return (libraw_kodak_makernotes_t)super.position(position);
      }
      @Override public libraw_kodak_makernotes_t getPointer(long i) {
          return new libraw_kodak_makernotes_t((Pointer)this).offsetAddress(i);
      }
  
    public native @Cast("ushort") short BlackLevelTop(); public native libraw_kodak_makernotes_t BlackLevelTop(short setter);
    public native @Cast("ushort") short BlackLevelBottom(); public native libraw_kodak_makernotes_t BlackLevelBottom(short setter);
    public native short offset_left(); public native libraw_kodak_makernotes_t offset_left(short setter);
    public native short offset_top(); public native libraw_kodak_makernotes_t offset_top(short setter); /* KDC files, negative values or zeros */
    public native @Cast("ushort") short clipBlack(); public native libraw_kodak_makernotes_t clipBlack(short setter);
    public native @Cast("ushort") short clipWhite(); public native libraw_kodak_makernotes_t clipWhite(short setter);   /* valid for P712, P850, P880 */
    public native float romm_camDaylight(int i, int j); public native libraw_kodak_makernotes_t romm_camDaylight(int i, int j, float setter);
    @MemberGetter public native @Cast("float(* /*[3]*/ )[3]") FloatPointer romm_camDaylight();
    public native float romm_camTungsten(int i, int j); public native libraw_kodak_makernotes_t romm_camTungsten(int i, int j, float setter);
    @MemberGetter public native @Cast("float(* /*[3]*/ )[3]") FloatPointer romm_camTungsten();
    public native float romm_camFluorescent(int i, int j); public native libraw_kodak_makernotes_t romm_camFluorescent(int i, int j, float setter);
    @MemberGetter public native @Cast("float(* /*[3]*/ )[3]") FloatPointer romm_camFluorescent();
    public native float romm_camFlash(int i, int j); public native libraw_kodak_makernotes_t romm_camFlash(int i, int j, float setter);
    @MemberGetter public native @Cast("float(* /*[3]*/ )[3]") FloatPointer romm_camFlash();
    public native float romm_camCustom(int i, int j); public native libraw_kodak_makernotes_t romm_camCustom(int i, int j, float setter);
    @MemberGetter public native @Cast("float(* /*[3]*/ )[3]") FloatPointer romm_camCustom();
    public native float romm_camAuto(int i, int j); public native libraw_kodak_makernotes_t romm_camAuto(int i, int j, float setter);
    @MemberGetter public native @Cast("float(* /*[3]*/ )[3]") FloatPointer romm_camAuto();
    public native @Cast("ushort") short val018percent(); public native libraw_kodak_makernotes_t val018percent(short setter);
    public native @Cast("ushort") short val100percent(); public native libraw_kodak_makernotes_t val100percent(short setter);
    public native @Cast("ushort") short val170percent(); public native libraw_kodak_makernotes_t val170percent(short setter);
    public native short MakerNoteKodak8a(); public native libraw_kodak_makernotes_t MakerNoteKodak8a(short setter);
    public native float ISOCalibrationGain(); public native libraw_kodak_makernotes_t ISOCalibrationGain(float setter);
    public native float AnalogISO(); public native libraw_kodak_makernotes_t AnalogISO(float setter);
  }