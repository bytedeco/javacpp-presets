// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.hdf5;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.hdf5.global.hdf5.*;


/**
 * \brief Information struct for links
 */
/** <!-- [H5L_info2_t_snip] --> */
@Properties(inherit = org.bytedeco.hdf5.presets.hdf5.class)
public class H5L_info2_t extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public H5L_info2_t() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public H5L_info2_t(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public H5L_info2_t(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public H5L_info2_t position(long position) {
        return (H5L_info2_t)super.position(position);
    }
    @Override public H5L_info2_t getPointer(long i) {
        return new H5L_info2_t((Pointer)this).offsetAddress(i);
    }

    /** Type of link                   */
    public native @Cast("H5L_type_t") int type(); public native H5L_info2_t type(int setter);
    /** Indicate if creation order is valid */
    public native @Cast("hbool_t") boolean corder_valid(); public native H5L_info2_t corder_valid(boolean setter);
    /** Creation order                 */
    public native @Cast("int64_t") long corder(); public native H5L_info2_t corder(long setter);
    /** Character set of link name     */
    public native @Cast("H5T_cset_t") int cset(); public native H5L_info2_t cset(int setter);
        /** Token of location that hard link points to */
        @Name("u.token") public native @ByRef H5O_token_t u_token(); public native H5L_info2_t u_token(H5O_token_t setter);
        /** Size of a soft link or user-defined link value */
        @Name("u.val_size") public native @Cast("size_t") long u_val_size(); public native H5L_info2_t u_val_size(long setter);
}
