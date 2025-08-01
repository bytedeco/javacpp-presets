// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.gsl;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.gsl.global.gsl.*;


/* Driver object
 *
 * This is a high level wrapper for step, control and
 * evolve objects. 
 */

@Name("gsl_odeiv2_driver_struct") @Properties(inherit = org.bytedeco.gsl.presets.gsl.class)
public class gsl_odeiv2_driver extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public gsl_odeiv2_driver() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public gsl_odeiv2_driver(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public gsl_odeiv2_driver(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public gsl_odeiv2_driver position(long position) {
        return (gsl_odeiv2_driver)super.position(position);
    }
    @Override public gsl_odeiv2_driver getPointer(long i) {
        return new gsl_odeiv2_driver((Pointer)this).offsetAddress(i);
    }

  public native @Const gsl_odeiv2_system sys(); public native gsl_odeiv2_driver sys(gsl_odeiv2_system setter); /* ODE system */
  public native gsl_odeiv2_step s(); public native gsl_odeiv2_driver s(gsl_odeiv2_step setter);           /* stepper object */
  public native gsl_odeiv2_control c(); public native gsl_odeiv2_driver c(gsl_odeiv2_control setter);        /* control object */
  public native gsl_odeiv2_evolve e(); public native gsl_odeiv2_driver e(gsl_odeiv2_evolve setter);         /* evolve object */
  public native double h(); public native gsl_odeiv2_driver h(double setter);                     /* step size */
  public native double hmin(); public native gsl_odeiv2_driver hmin(double setter);                  /* minimum step size allowed */
  public native double hmax(); public native gsl_odeiv2_driver hmax(double setter);                  /* maximum step size allowed */
  public native @Cast("unsigned long int") long n(); public native gsl_odeiv2_driver n(long setter);          /* number of steps taken */
  public native @Cast("unsigned long int") long nmax(); public native gsl_odeiv2_driver nmax(long setter);       /* Maximum number of steps allowed */
}
