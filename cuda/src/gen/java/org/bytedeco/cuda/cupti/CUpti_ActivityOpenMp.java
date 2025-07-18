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
 * \brief The base activity record for OpenMp records.
 *
 * @see CUpti_ActivityKind
 */
@Properties(inherit = org.bytedeco.cuda.presets.cupti.class)
public class CUpti_ActivityOpenMp extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CUpti_ActivityOpenMp() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CUpti_ActivityOpenMp(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CUpti_ActivityOpenMp(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CUpti_ActivityOpenMp position(long position) {
        return (CUpti_ActivityOpenMp)super.position(position);
    }
    @Override public CUpti_ActivityOpenMp getPointer(long i) {
        return new CUpti_ActivityOpenMp((Pointer)this).offsetAddress(i);
    }


  /**
   * The kind of this activity.
   */
  public native @Cast("CUpti_ActivityKind") int kind(); public native CUpti_ActivityOpenMp kind(int setter);

  /**
   * CUPTI OpenMP event kind (@see CUpti_OpenMpEventKind)
   */
  public native @Cast("CUpti_OpenMpEventKind") int eventKind(); public native CUpti_ActivityOpenMp eventKind(int setter);

  /**
   * Version number
   */
  public native @Cast("uint32_t") int version(); public native CUpti_ActivityOpenMp version(int setter);

  /**
   * ThreadId
   */
  public native @Cast("uint32_t") int threadId(); public native CUpti_ActivityOpenMp threadId(int setter);

  /**
   * CUPTI start timestamp
   */
  public native @Cast("uint64_t") long start(); public native CUpti_ActivityOpenMp start(long setter);

  /**
   * CUPTI end timestamp
   */
  public native @Cast("uint64_t") long end(); public native CUpti_ActivityOpenMp end(long setter);

  /**
   * The ID of the process where the OpenMP activity is executing.
   */
  public native @Cast("uint32_t") int cuProcessId(); public native CUpti_ActivityOpenMp cuProcessId(int setter);

  /**
   * The ID of the thread where the OpenMP activity is executing.
   */
  public native @Cast("uint32_t") int cuThreadId(); public native CUpti_ActivityOpenMp cuThreadId(int setter);
}
