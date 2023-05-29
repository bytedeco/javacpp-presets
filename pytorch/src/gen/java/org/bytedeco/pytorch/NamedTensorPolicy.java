// Targeted by JavaCPP version 1.5.9-SNAPSHOT: DO NOT EDIT THIS FILE

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

@Name("torch::jit::detail::NamedPolicy<torch::jit::detail::ParameterPolicy>") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NamedTensorPolicy extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NamedTensorPolicy() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public NamedTensorPolicy(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NamedTensorPolicy(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public NamedTensorPolicy position(long position) {
        return (NamedTensorPolicy)super.position(position);
    }
    @Override public NamedTensorPolicy getPointer(long i) {
        return new NamedTensorPolicy((Pointer)this).offsetAddress(i);
    }

  public static native @ByVal @Cast("torch::jit::detail::NamedPolicy<torch::jit::detail::ParameterPolicy>::value_type*") NamedJitModule create(
        @StdVector SlotCursor cursors,
        @ByVal IValue v);
  public static native @Cast("bool") boolean valid(@Const @SharedPtr("c10::ClassType") @ByRef ClassType t, @Cast("size_t") long i, @Const @ByRef IValue v);
  @MemberGetter public static native @Cast("const bool") boolean all_slots();
  public static final boolean all_slots = all_slots();
}