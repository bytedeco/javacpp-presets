// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.dnnl;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.dnnl.global.dnnl.*;


/** \} dnnl_api_shuffle
 <p>
 *  \addtogroup dnnl_api_binary Binary
 * 
 *  A primitive to perform tensor operations over two tensors.
 * 
 *  @see \ref dev_guide_binary in developer guide
 * 
 *  \{
 <p>
 *  Elementwise binary operator primitive. */
@Namespace("dnnl") @Properties(inherit = org.bytedeco.dnnl.presets.dnnl.class)
public class binary extends primitive {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public binary(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public binary(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public binary position(long position) {
        return (binary)super.position(position);
    }
    @Override public binary getPointer(long i) {
        return new binary(this).position(position + i);
    }

    /** Descriptor for an elementwise binary operator primitive. */
    @NoOffset public static class desc extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public desc(Pointer p) { super(p); }
        /** Native array allocator. Access with {@link Pointer#position(long)}. */
        public desc(long size) { super((Pointer)null); allocateArray(size); }
        private native void allocateArray(long size);
        @Override public desc position(long position) {
            return (desc)super.position(position);
        }
        @Override public desc getPointer(long i) {
            return new desc(this).position(position + i);
        }
    
        /** Underlying C operation descriptor. */
        public native @ByRef dnnl_binary_desc_t data(); public native desc data(dnnl_binary_desc_t setter);

        /** Default constructor. Produces an empty object. */
        
        ///
        public desc() { super((Pointer)null); allocate(); }
        private native void allocate();

        /** Constructs a descriptor for an elementwise binary operator
         *  primitive.
         * 
         *  @param aalgorithm Elementwise binary algorithm.
         *  @param src0 Memory descriptor for source tensor #0.
         *  @param src1 Memory descriptor for source tensor #1.
         *  @param dst Memory descriptor for destination tensor. */
        public desc(algorithm aalgorithm, @Const @ByRef memory.desc src0,
                        @Const @ByRef memory.desc src1, @Const @ByRef memory.desc dst) { super((Pointer)null); allocate(aalgorithm, src0, src1, dst); }
        private native void allocate(algorithm aalgorithm, @Const @ByRef memory.desc src0,
                        @Const @ByRef memory.desc src1, @Const @ByRef memory.desc dst);
        public desc(@Cast("dnnl::algorithm") int aalgorithm, @Const @ByRef memory.desc src0,
                        @Const @ByRef memory.desc src1, @Const @ByRef memory.desc dst) { super((Pointer)null); allocate(aalgorithm, src0, src1, dst); }
        private native void allocate(@Cast("dnnl::algorithm") int aalgorithm, @Const @ByRef memory.desc src0,
                        @Const @ByRef memory.desc src1, @Const @ByRef memory.desc dst);
    }

    /** Primitive descriptor for an elementwise binary operator primitive. */
    public static class primitive_desc extends org.bytedeco.dnnl.primitive_desc {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public primitive_desc(Pointer p) { super(p); }
        /** Native array allocator. Access with {@link Pointer#position(long)}. */
        public primitive_desc(long size) { super((Pointer)null); allocateArray(size); }
        private native void allocateArray(long size);
        @Override public primitive_desc position(long position) {
            return (primitive_desc)super.position(position);
        }
        @Override public primitive_desc getPointer(long i) {
            return new primitive_desc(this).position(position + i);
        }
    
        /** Default constructor. Produces an empty object. */
        
        ///
        public primitive_desc() { super((Pointer)null); allocate(); }
        private native void allocate();

        /** Constructs a primitive descriptor for an elementwise binary operator
         *  primitive.
         * 
         *  @param adesc Descriptor for an elementwise binary operator primitive.
         *  @param aengine Engine to use.
         *  @param allow_empty A flag signifying whether construction is
         *      allowed to fail without throwing an exception. In this case an
         *      empty object will be produced. This flag is optional and
         *      defaults to false. */
        
        ///
        public primitive_desc(@Const @ByRef desc adesc, @Const @ByRef engine aengine,
                        @Cast("bool") boolean allow_empty/*=false*/) { super((Pointer)null); allocate(adesc, aengine, allow_empty); }
        private native void allocate(@Const @ByRef desc adesc, @Const @ByRef engine aengine,
                        @Cast("bool") boolean allow_empty/*=false*/);
        public primitive_desc(@Const @ByRef desc adesc, @Const @ByRef engine aengine) { super((Pointer)null); allocate(adesc, aengine); }
        private native void allocate(@Const @ByRef desc adesc, @Const @ByRef engine aengine);

        /** Constructs a primitive descriptor for an elementwise binary operator
         *  primitive.
         * 
         *  @param adesc Descriptor for an elementwise binary operator primitive.
         *  @param aengine Engine to use.
         *  @param attr Primitive attributes to use.
         *  @param allow_empty A flag signifying whether construction is
         *      allowed to fail without throwing an exception. In this case an
         *      empty object will be produced. This flag is optional and
         *      defaults to false. */
        
        ///
        public primitive_desc(@Const @ByRef desc adesc, @Const @ByRef primitive_attr attr,
                        @Const @ByRef engine aengine, @Cast("bool") boolean allow_empty/*=false*/) { super((Pointer)null); allocate(adesc, attr, aengine, allow_empty); }
        private native void allocate(@Const @ByRef desc adesc, @Const @ByRef primitive_attr attr,
                        @Const @ByRef engine aengine, @Cast("bool") boolean allow_empty/*=false*/);
        public primitive_desc(@Const @ByRef desc adesc, @Const @ByRef primitive_attr attr,
                        @Const @ByRef engine aengine) { super((Pointer)null); allocate(adesc, attr, aengine); }
        private native void allocate(@Const @ByRef desc adesc, @Const @ByRef primitive_attr attr,
                        @Const @ByRef engine aengine);

        /** Constructs a primitive descriptor for a binary primitive from a C
         *  API primitive descriptor that must have a matching kind.
         * 
         *  @param pd C API primitive descriptor for a binary primitive. */
        public primitive_desc(dnnl_primitive_desc pd) { super((Pointer)null); allocate(pd); }
        private native void allocate(dnnl_primitive_desc pd);

        /** \copydoc dnnl::primitive_desc_base::src_desc(int)const */
        public native @ByVal memory.desc src_desc(int idx/*=0*/);
        public native @ByVal memory.desc src_desc();

        /** Returns the memory descriptor for source #0. */
        public native @ByVal memory.desc src0_desc();

        /** Returns the memory descriptor for source #1. */
        public native @ByVal memory.desc src1_desc();

        /** \copydoc dnnl::primitive_desc_base::dst_desc()const */
        public native @ByVal memory.desc dst_desc();
    }

    /** Default constructor. Produces an empty object. */
    public binary() { super((Pointer)null); allocate(); }
    private native void allocate();

    /** Constructs an elementwise binary operation primitive.
     *  @param pd Primitive descriptor for an elementwise binary operation
     *      primitive. */
    public binary(@Const @ByRef primitive_desc pd) { super((Pointer)null); allocate(pd); }
    private native void allocate(@Const @ByRef primitive_desc pd);
}