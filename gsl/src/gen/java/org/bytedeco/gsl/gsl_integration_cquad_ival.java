// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.gsl;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.gsl.global.gsl.*;



/* Cquad integration - Pedro Gonnet */

/* Data of a single interval */
@Properties(inherit = org.bytedeco.gsl.presets.gsl.class)
public class gsl_integration_cquad_ival extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public gsl_integration_cquad_ival() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public gsl_integration_cquad_ival(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public gsl_integration_cquad_ival(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public gsl_integration_cquad_ival position(long position) {
        return (gsl_integration_cquad_ival)super.position(position);
    }
    @Override public gsl_integration_cquad_ival getPointer(long i) {
        return new gsl_integration_cquad_ival((Pointer)this).offsetAddress(i);
    }

  public native double a(); public native gsl_integration_cquad_ival a(double setter);
  public native double b(); public native gsl_integration_cquad_ival b(double setter);
  public native double c(int i); public native gsl_integration_cquad_ival c(int i, double setter);
  @MemberGetter public native DoublePointer c();
  public native double fx(int i); public native gsl_integration_cquad_ival fx(int i, double setter);
  @MemberGetter public native DoublePointer fx();
  public native double igral(); public native gsl_integration_cquad_ival igral(double setter);
  public native double err(); public native gsl_integration_cquad_ival err(double setter);
  public native int depth(); public native gsl_integration_cquad_ival depth(int setter);
  public native int rdepth(); public native gsl_integration_cquad_ival rdepth(int setter);
  public native int ndiv(); public native gsl_integration_cquad_ival ndiv(int setter);
}
