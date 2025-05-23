// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.hdf5;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.hdf5.global.hdf5.*;

/** <!-- [H5Z_func_t_snip] -->
<p>
/**
 * The filter table maps filter identification numbers to structs that
 * contain a pointers to the filter function and timing statistics.
 */
/** <!-- [H5Z_class2_t_snip] --> */
@Properties(inherit = org.bytedeco.hdf5.presets.hdf5.class)
public class H5Z_class2_t extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public H5Z_class2_t() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public H5Z_class2_t(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public H5Z_class2_t(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public H5Z_class2_t position(long position) {
        return (H5Z_class2_t)super.position(position);
    }
    @Override public H5Z_class2_t getPointer(long i) {
        return new H5Z_class2_t((Pointer)this).offsetAddress(i);
    }

    /** Version number of the H5Z_class_t struct     */
    public native int version(); public native H5Z_class2_t version(int setter);
    /** Filter ID number                             */
    public native @Cast("H5Z_filter_t") int id(); public native H5Z_class2_t id(int setter);
    /** Does this filter have an encoder?            */
    public native @Cast("unsigned") int encoder_present(); public native H5Z_class2_t encoder_present(int setter);
    /** Does this filter have a decoder?             */
    public native @Cast("unsigned") int decoder_present(); public native H5Z_class2_t decoder_present(int setter);
    /** Comment for debugging                        */
    public native @Cast("const char*") BytePointer name(); public native H5Z_class2_t name(BytePointer setter);
    /** The "can apply" callback for a filter        */
    public native H5Z_can_apply_func_t can_apply(); public native H5Z_class2_t can_apply(H5Z_can_apply_func_t setter);
    /** The "set local" callback for a filter        */
    public native H5Z_set_local_func_t set_local(); public native H5Z_class2_t set_local(H5Z_set_local_func_t setter);
    /** The actual filter function                   */
    public native H5Z_func_t filter(); public native H5Z_class2_t filter(H5Z_func_t setter);
}
