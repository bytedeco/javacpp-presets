// Targeted by JavaCPP version 1.5.10-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.pytorch;

import org.bytedeco.pytorch.Allocator;
import org.bytedeco.pytorch.Function;
import org.bytedeco.pytorch.functions.*;
import org.bytedeco.pytorch.Module;
import org.bytedeco.javacpp.annotation.Cast;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.pytorch.global.torch.*;


@Namespace("ska_ordered") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class power_of_two_hash_policy extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public power_of_two_hash_policy() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public power_of_two_hash_policy(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public power_of_two_hash_policy(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public power_of_two_hash_policy position(long position) {
        return (power_of_two_hash_policy)super.position(position);
    }
    @Override public power_of_two_hash_policy getPointer(long i) {
        return new power_of_two_hash_policy((Pointer)this).offsetAddress(i);
    }

  public native @Cast("uint64_t") long index_for_hash(@Cast("uint64_t") long hash, @Cast("uint64_t") long num_slots_minus_one);
  public native @Cast("uint64_t") long keep_in_range(@Cast("uint64_t") long index, @Cast("uint64_t") long num_slots_minus_one);
  public native byte next_size_over(@Cast("uint64_t*") @ByRef LongPointer size);
  public native byte next_size_over(@Cast("uint64_t*") @ByRef LongBuffer size);
  public native byte next_size_over(@Cast("uint64_t*") @ByRef long[] size);
  public native void commit(byte arg0);
  public native void reset();
}