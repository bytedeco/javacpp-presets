// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cupti;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.cuda.global.cupti.*;


/**
 * \brief The activity record for CUDA stream.
 *
 * This activity is used to track created streams.
 * (CUPTI_ACTIVITY_KIND_STREAM).
 */
@Properties(inherit = org.bytedeco.cuda.presets.cupti.class)
public class CUpti_ActivityStream extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CUpti_ActivityStream() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CUpti_ActivityStream(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CUpti_ActivityStream(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CUpti_ActivityStream position(long position) {
        return (CUpti_ActivityStream)super.position(position);
    }
    @Override public CUpti_ActivityStream getPointer(long i) {
        return new CUpti_ActivityStream((Pointer)this).offsetAddress(i);
    }

  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_STREAM.
   */
  public native @Cast("CUpti_ActivityKind") int kind(); public native CUpti_ActivityStream kind(int setter);
  /**
   * The ID of the context where the stream was created.
   */
  public native @Cast("uint32_t") int contextId(); public native CUpti_ActivityStream contextId(int setter);

  /**
   * A unique stream ID to identify the stream.
   */
  public native @Cast("uint32_t") int streamId(); public native CUpti_ActivityStream streamId(int setter);

  /**
   * The clamped priority for the stream.
   */
  public native @Cast("uint32_t") int priority(); public native CUpti_ActivityStream priority(int setter);

  /**
   * Flags associated with the stream.
   */
  public native @Cast("CUpti_ActivityStreamFlag") int flag(); public native CUpti_ActivityStream flag(int setter);

  /**
   * The correlation ID of the API to which this result is associated.
   */
  public native @Cast("uint32_t") int correlationId(); public native CUpti_ActivityStream correlationId(int setter);
}
