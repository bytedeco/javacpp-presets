// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cufile;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.cuda.global.cufile.*;

@Properties(inherit = org.bytedeco.cuda.presets.cufile.class)
public class CUfileIOParams_t extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CUfileIOParams_t() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CUfileIOParams_t(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CUfileIOParams_t(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CUfileIOParams_t position(long position) {
        return (CUfileIOParams_t)super.position(position);
    }
    @Override public CUfileIOParams_t getPointer(long i) {
        return new CUfileIOParams_t((Pointer)this).offsetAddress(i);
    }

	public native @Cast("CUfileBatchMode_t") int mode(); public native CUfileIOParams_t mode(int setter); // Must be the very first field.
			@Name("u.batch.devPtr_base") public native Pointer u_batch_devPtr_base(); public native CUfileIOParams_t u_batch_devPtr_base(Pointer setter); //This can be a device memory or a host memory pointer.
			@Name("u.batch.file_offset") public native @Cast("off_t") long u_batch_file_offset(); public native CUfileIOParams_t u_batch_file_offset(long setter);
			@Name("u.batch.devPtr_offset") public native @Cast("off_t") long u_batch_devPtr_offset(); public native CUfileIOParams_t u_batch_devPtr_offset(long setter); 
			@Name("u.batch.size") public native @Cast("size_t") long u_batch_size(); public native CUfileIOParams_t u_batch_size(long setter);
	public native CUfileHandle_t fh(); public native CUfileIOParams_t fh(CUfileHandle_t setter);
	public native @Cast("CUfileOpcode_t") int opcode(); public native CUfileIOParams_t opcode(int setter);
	public native Pointer cookie(); public native CUfileIOParams_t cookie(Pointer setter);
}
