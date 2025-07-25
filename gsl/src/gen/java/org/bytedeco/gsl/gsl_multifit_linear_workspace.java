// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.gsl;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.gsl.global.gsl.*;

// #else
// #endif

@Properties(inherit = org.bytedeco.gsl.presets.gsl.class)
public class gsl_multifit_linear_workspace extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public gsl_multifit_linear_workspace() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public gsl_multifit_linear_workspace(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public gsl_multifit_linear_workspace(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public gsl_multifit_linear_workspace position(long position) {
        return (gsl_multifit_linear_workspace)super.position(position);
    }
    @Override public gsl_multifit_linear_workspace getPointer(long i) {
        return new gsl_multifit_linear_workspace((Pointer)this).offsetAddress(i);
    }

  public native @Cast("size_t") long nmax(); public native gsl_multifit_linear_workspace nmax(long setter);         /* maximum number of observations */
  public native @Cast("size_t") long pmax(); public native gsl_multifit_linear_workspace pmax(long setter);         /* maximum number of parameters */
  public native @Cast("size_t") long n(); public native gsl_multifit_linear_workspace n(long setter);            /* number of observations in current SVD decomposition */
  public native @Cast("size_t") long p(); public native gsl_multifit_linear_workspace p(long setter);            /* number of parameters in current SVD decomposition */
  public native gsl_matrix A(); public native gsl_multifit_linear_workspace A(gsl_matrix setter);      /* least squares matrix for SVD, n-by-p */
  public native gsl_matrix Q(); public native gsl_multifit_linear_workspace Q(gsl_matrix setter);
  public native gsl_matrix QSI(); public native gsl_multifit_linear_workspace QSI(gsl_matrix setter);
  public native gsl_vector S(); public native gsl_multifit_linear_workspace S(gsl_vector setter);
  public native gsl_vector t(); public native gsl_multifit_linear_workspace t(gsl_vector setter);
  public native gsl_vector xt(); public native gsl_multifit_linear_workspace xt(gsl_vector setter);
  public native gsl_vector D(); public native gsl_multifit_linear_workspace D(gsl_vector setter);
  public native double rcond(); public native gsl_multifit_linear_workspace rcond(double setter);        /* reciprocal condition number */
}
