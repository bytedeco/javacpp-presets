// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.pytorch;

import org.bytedeco.pytorch.Allocator;
import org.bytedeco.pytorch.Function;
import org.bytedeco.pytorch.Module;
import org.bytedeco.javacpp.annotation.Cast;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;
import org.bytedeco.javacpp.chrono.*;
import static org.bytedeco.javacpp.global.chrono.*;

import static org.bytedeco.pytorch.global.torch.*;


// Thread local debug information is propagated across the forward
// (including async fork tasks) and backward passes and is supposed
// to be utilized by the user's code to pass extra information from
// the higher layers (e.g. model id) down to the lower levels
// (e.g. to the operator observers used for debugging, logging,
// profiling, etc)
@Namespace("c10") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ThreadLocalDebugInfo extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ThreadLocalDebugInfo() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public ThreadLocalDebugInfo(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ThreadLocalDebugInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public ThreadLocalDebugInfo position(long position) {
        return (ThreadLocalDebugInfo)super.position(position);
    }
    @Override public ThreadLocalDebugInfo getPointer(long i) {
        return new ThreadLocalDebugInfo((Pointer)this).offsetAddress(i);
    }

  public static native DebugInfoBase get(DebugInfoKind kind);
  public static native DebugInfoBase get(@Cast("c10::DebugInfoKind") byte kind);

  // Get current ThreadLocalDebugInfo
  public static native @SharedPtr ThreadLocalDebugInfo current();

  // Internal, use DebugInfoGuard/ThreadLocalStateGuard
  

  // Push debug info struct of a given kind
  public static native void _push(DebugInfoKind kind, @SharedPtr DebugInfoBase info);
  public static native void _push(@Cast("c10::DebugInfoKind") byte kind, @SharedPtr DebugInfoBase info);
  // Pop debug info, throws in case the last pushed
  // debug info is not of a given kind
  public static native @SharedPtr DebugInfoBase _pop(DebugInfoKind kind);
  public static native @SharedPtr DebugInfoBase _pop(@Cast("c10::DebugInfoKind") byte kind);
  // Peek debug info, throws in case the last pushed debug info is not of the
  // given kind
  public static native @SharedPtr DebugInfoBase _peek(DebugInfoKind kind);
  public static native @SharedPtr DebugInfoBase _peek(@Cast("c10::DebugInfoKind") byte kind);
}
