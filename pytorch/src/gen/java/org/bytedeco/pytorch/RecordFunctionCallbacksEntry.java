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


// It is unnecessary to use atomic operations for enabling
// thread-local function callbacks. Moreover, it prevents saving to
// ThreadLocalState because std::atomic is non-copyable.
@Namespace("at") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RecordFunctionCallbacksEntry extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public RecordFunctionCallbacksEntry(Pointer p) { super(p); }

  public RecordFunctionCallbacksEntry(@ByVal @Cast("at::RecordFunctionCallback*") Pointer cb, @Cast("at::CallbackHandle") long h) { super((Pointer)null); allocate(cb, h); }
  private native void allocate(@ByVal @Cast("at::RecordFunctionCallback*") Pointer cb, @Cast("at::CallbackHandle") long h);

  public native @ByRef @Cast("at::RecordFunctionCallback*") Pointer callback_(); public native RecordFunctionCallbacksEntry callback_(Pointer setter);
  public native @Cast("bool") boolean enabled_(); public native RecordFunctionCallbacksEntry enabled_(boolean setter);
  public native @Cast("at::CallbackHandle") long handle_(); public native RecordFunctionCallbacksEntry handle_(long setter);
}
