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
 * \brief The activity record for memory copies.
 *
 * This activity record represents a memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY).
 */
@Properties(inherit = org.bytedeco.cuda.presets.cupti.class)
public class CUpti_ActivityMemcpy6 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CUpti_ActivityMemcpy6() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CUpti_ActivityMemcpy6(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CUpti_ActivityMemcpy6(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CUpti_ActivityMemcpy6 position(long position) {
        return (CUpti_ActivityMemcpy6)super.position(position);
    }
    @Override public CUpti_ActivityMemcpy6 getPointer(long i) {
        return new CUpti_ActivityMemcpy6((Pointer)this).offsetAddress(i);
    }

  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY.
   */
  public native @Cast("CUpti_ActivityKind") int kind(); public native CUpti_ActivityMemcpy6 kind(int setter);

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. @see CUpti_ActivityMemcpyKind
   */
  public native @Cast("uint8_t") byte copyKind(); public native CUpti_ActivityMemcpy6 copyKind(byte setter);

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. @see CUpti_ActivityMemoryKind
   */
  public native @Cast("uint8_t") byte srcKind(); public native CUpti_ActivityMemcpy6 srcKind(byte setter);

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. @see CUpti_ActivityMemoryKind
   */
  public native @Cast("uint8_t") byte dstKind(); public native CUpti_ActivityMemcpy6 dstKind(byte setter);

  /**
   * The flags associated with the memory copy. @see CUpti_ActivityFlag
   */
  public native @Cast("uint8_t") byte flags(); public native CUpti_ActivityMemcpy6 flags(byte setter);

  /**
   * The number of bytes transferred by the memory copy.
   */
  public native @Cast("uint64_t") long bytes(); public native CUpti_ActivityMemcpy6 bytes(long setter);

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  public native @Cast("uint64_t") long start(); public native CUpti_ActivityMemcpy6 start(long setter);

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  public native @Cast("uint64_t") long end(); public native CUpti_ActivityMemcpy6 end(long setter);

  /**
   * The ID of the device where the memory copy is occurring.
   */
  public native @Cast("uint32_t") int deviceId(); public native CUpti_ActivityMemcpy6 deviceId(int setter);

  /**
   * The ID of the context where the memory copy is occurring.
   */
  public native @Cast("uint32_t") int contextId(); public native CUpti_ActivityMemcpy6 contextId(int setter);

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  public native @Cast("uint32_t") int streamId(); public native CUpti_ActivityMemcpy6 streamId(int setter);

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  public native @Cast("uint32_t") int correlationId(); public native CUpti_ActivityMemcpy6 correlationId(int setter);

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  public native @Cast("uint32_t") int runtimeCorrelationId(); public native CUpti_ActivityMemcpy6 runtimeCorrelationId(int setter);

// #ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  public native @Cast("uint32_t") int pad(); public native CUpti_ActivityMemcpy6 pad(int setter);
// #endif

  /**
   * Undefined. Reserved for internal use.
   */
  public native Pointer reserved0(); public native CUpti_ActivityMemcpy6 reserved0(Pointer setter);

  /**
   * The unique ID of the graph node that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  public native @Cast("uint64_t") long graphNodeId(); public native CUpti_ActivityMemcpy6 graphNodeId(long setter);

  /**
   * The unique ID of the graph that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  public native @Cast("uint32_t") int graphId(); public native CUpti_ActivityMemcpy6 graphId(int setter);

  /**
   * The ID of the HW channel on which the memory copy is occurring.
   */
  public native @Cast("uint32_t") int channelID(); public native CUpti_ActivityMemcpy6 channelID(int setter);

  /**
   * The type of the channel
   */
  public native @Cast("CUpti_ChannelType") int channelType(); public native CUpti_ActivityMemcpy6 channelType(int setter);

  /**
   *  Reserved for internal use.
   */
  public native @Cast("uint32_t") int pad2(); public native CUpti_ActivityMemcpy6 pad2(int setter);

  /**
   * The total number of memcopy operations traced in this record.
   * This field is valid for memcpy operations happening using
   * MemcpyBatchAsync APIs in CUDA.
   * In MemcpyBatchAsync APIs, multiple memcpy operations are batched
   * together for optimization purposes based on certain heuristics.
   * For other memcpy operations, this field will be 1.
   */
   public native @Cast("uint64_t") long copyCount(); public native CUpti_ActivityMemcpy6 copyCount(long setter);
}
