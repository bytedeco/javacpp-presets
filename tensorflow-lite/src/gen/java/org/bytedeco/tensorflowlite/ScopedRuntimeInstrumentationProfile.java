// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tensorflowlite;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;


// Similar to ScopedProfile but has extra event metadata for EndEvent.
@Namespace("tflite") @NoOffset @Properties(inherit = org.bytedeco.tensorflowlite.presets.tensorflowlite.class)
public class ScopedRuntimeInstrumentationProfile extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ScopedRuntimeInstrumentationProfile(Pointer p) { super(p); }

  public ScopedRuntimeInstrumentationProfile(Profiler profiler, @Cast("const char*") BytePointer tag) { super((Pointer)null); allocate(profiler, tag); }
  private native void allocate(Profiler profiler, @Cast("const char*") BytePointer tag);
  public ScopedRuntimeInstrumentationProfile(Profiler profiler, String tag) { super((Pointer)null); allocate(profiler, tag); }
  private native void allocate(Profiler profiler, String tag);

  public native void set_runtime_status(@Cast("int64_t") long delegate_status, @Cast("int64_t") long interpreter_status);
}
