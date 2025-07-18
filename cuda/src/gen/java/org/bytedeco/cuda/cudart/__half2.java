// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cudart;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cuda.global.cudart.*;

// #endif /* !defined(__CUDA_NO_HALF_OPERATORS__) */

/**
 * \ingroup CUDA_MATH_INTRINSIC_HALF
 * \brief __half2 data type
 * \details This structure implements the datatype for storing two 
 * half-precision floating-point numbers. 
 * The structure implements assignment, arithmetic and comparison
 * operators, and type conversions. 
 * 
 * - NOTE: __half2 is visible to non-nvcc host compilers
 */
@NoOffset @Properties(inherit = org.bytedeco.cuda.presets.cudart.class)
public class __half2 extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public __half2(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public __half2(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public __half2 position(long position) {
        return (__half2)super.position(position);
    }
    @Override public __half2 getPointer(long i) {
        return new __half2((Pointer)this).offsetAddress(i);
    }

    /**
     * Storage field holding lower \p __half part.
     */
    public native @ByRef __half x(); public native __half2 x(__half setter);
    /**
     * Storage field holding upper \p __half part.
     */
    public native @ByRef __half y(); public native __half2 y(__half setter);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * \brief Constructor by default.
     * \details Emtpy default constructor, result is uninitialized.
     */
// #if defined(__CPP_VERSION_AT_LEAST_11_FP16)
    public __half2() { super((Pointer)null); allocate(); }
    private native void allocate();
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Move constructor, available for \p C++11 and later dialects
     */
    public __half2(@Const @ByRef(true) __half2 src) { super((Pointer)null); allocate(src); }
    private native void allocate(@Const @ByRef(true) __half2 src);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Move assignment operator, available for \p C++11 and later dialects
     */
    public native @ByRef @Name("operator =") __half2 put(@Const @ByRef(true) __half2 src);
// #else
// #endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */

    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Constructor from two \p __half variables
     */
    public __half2(@Const @ByRef __half a, @Const @ByRef __half b) { super((Pointer)null); allocate(a, b); }
    private native void allocate(@Const @ByRef __half a, @Const @ByRef __half b);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Copy constructor
     */    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Copy assignment operator
     */

    /* Convert to/from __half2_raw */
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Constructor from \p __half2_raw
     */
    public __half2(@Const @ByRef __half2_raw h2r ) { super((Pointer)null); allocate(h2r); }
    private native void allocate(@Const @ByRef __half2_raw h2r );
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Assignment operator from \p __half2_raw
     */
    public native @ByRef @Name("operator =") __half2 put(@Const @ByRef __half2_raw h2r);
    /**
     * \ingroup CUDA_MATH__HALF_MISC
     * Conversion operator to \p __half2_raw
     */
    public native @ByVal @Name("operator __half2_raw") __half2_raw as__half2_raw();
}
