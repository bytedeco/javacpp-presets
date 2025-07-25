// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tritonserver.tritondevelopertoolsserver;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tritonserver.global.tritondevelopertoolsserver.*;


//==============================================================================
// TritonException
//
@Namespace("triton::developer_tools::server") @NoOffset @Properties(inherit = org.bytedeco.tritonserver.presets.tritondevelopertoolsserver.class)
public class TritonException extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TritonException(Pointer p) { super(p); }

  public TritonException(@StdString BytePointer message) { super((Pointer)null); allocate(message); }
  private native void allocate(@StdString BytePointer message);
  public TritonException(@StdString String message) { super((Pointer)null); allocate(message); }
  private native void allocate(@StdString String message);

  public native String what();

  public native @StdString BytePointer message_(); public native TritonException message_(BytePointer setter);
}
