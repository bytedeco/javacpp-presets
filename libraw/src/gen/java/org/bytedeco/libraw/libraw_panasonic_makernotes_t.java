// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.libraw;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.libraw.global.LibRaw.*;


  @Properties(inherit = org.bytedeco.libraw.presets.LibRaw.class)
public class libraw_panasonic_makernotes_t extends Pointer {
      static { Loader.load(); }
      /** Default native constructor. */
      public libraw_panasonic_makernotes_t() { super((Pointer)null); allocate(); }
      /** Native array allocator. Access with {@link Pointer#position(long)}. */
      public libraw_panasonic_makernotes_t(long size) { super((Pointer)null); allocateArray(size); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public libraw_panasonic_makernotes_t(Pointer p) { super(p); }
      private native void allocate();
      private native void allocateArray(long size);
      @Override public libraw_panasonic_makernotes_t position(long position) {
          return (libraw_panasonic_makernotes_t)super.position(position);
      }
      @Override public libraw_panasonic_makernotes_t getPointer(long i) {
          return new libraw_panasonic_makernotes_t((Pointer)this).offsetAddress(i);
      }
  
    /* Compression:
     34826 (Panasonic RAW 2): LEICA DIGILUX 2;
     34828 (Panasonic RAW 3): LEICA D-LUX 3; LEICA V-LUX 1; Panasonic DMC-LX1;
     Panasonic DMC-LX2; Panasonic DMC-FZ30; Panasonic DMC-FZ50; 34830 (not in
     exiftool): LEICA DIGILUX 3; Panasonic DMC-L1; 34316 (Panasonic RAW 1):
     others (LEICA, Panasonic, YUNEEC);
    */
    public native @Cast("ushort") short Compression(); public native libraw_panasonic_makernotes_t Compression(short setter);
    public native @Cast("ushort") short BlackLevelDim(); public native libraw_panasonic_makernotes_t BlackLevelDim(short setter);
    public native float BlackLevel(int i); public native libraw_panasonic_makernotes_t BlackLevel(int i, float setter);
    @MemberGetter public native FloatPointer BlackLevel();
    public native @Cast("unsigned") int Multishot(); public native libraw_panasonic_makernotes_t Multishot(int setter); /* 0 is Off, 65536 is Pixel Shift */
    public native float gamma(); public native libraw_panasonic_makernotes_t gamma(float setter);
    public native int HighISOMultiplier(int i); public native libraw_panasonic_makernotes_t HighISOMultiplier(int i, int setter);
    @MemberGetter public native IntPointer HighISOMultiplier(); /* 0->R, 1->G, 2->B */
    public native short FocusStepNear(); public native libraw_panasonic_makernotes_t FocusStepNear(short setter);
    public native short FocusStepCount(); public native libraw_panasonic_makernotes_t FocusStepCount(short setter);
    public native @Cast("unsigned") int ZoomPosition(); public native libraw_panasonic_makernotes_t ZoomPosition(int setter);
    public native @Cast("unsigned") int LensManufacturer(); public native libraw_panasonic_makernotes_t LensManufacturer(int setter);
  }
