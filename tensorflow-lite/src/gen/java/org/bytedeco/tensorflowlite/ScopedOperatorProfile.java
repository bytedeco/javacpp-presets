// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tensorflowlite;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;


@Namespace("tflite") @Properties(inherit = org.bytedeco.tensorflowlite.presets.tensorflowlite.class)
public class ScopedOperatorProfile extends ScopedProfile {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ScopedOperatorProfile(Pointer p) { super(p); }

  public ScopedOperatorProfile(Profiler profiler, @Cast("const char*") BytePointer tag, int node_index) { super((Pointer)null); allocate(profiler, tag, node_index); }
  private native void allocate(Profiler profiler, @Cast("const char*") BytePointer tag, int node_index);
  public ScopedOperatorProfile(Profiler profiler, String tag, int node_index) { super((Pointer)null); allocate(profiler, tag, node_index); }
  private native void allocate(Profiler profiler, String tag, int node_index);
}
