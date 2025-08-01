// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.onnxruntime;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.opencl.*;
import static org.bytedeco.opencl.global.OpenCL.*;
import org.bytedeco.dnnl.*;
import static org.bytedeco.dnnl.global.dnnl.*;

import static org.bytedeco.onnxruntime.global.onnxruntime.*;

@Name("Ort::detail::ConstSessionOptionsImpl<OrtSessionOptions>") @Properties(inherit = org.bytedeco.onnxruntime.presets.onnxruntime.class)
public class ConstSessionOptionsImpl extends BaseSessionOptions {
    static { Loader.load(); }
    /** Default native constructor. */
    public ConstSessionOptionsImpl() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public ConstSessionOptionsImpl(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ConstSessionOptionsImpl(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public ConstSessionOptionsImpl position(long position) {
        return (ConstSessionOptionsImpl)super.position(position);
    }
    @Override public ConstSessionOptionsImpl getPointer(long i) {
        return new ConstSessionOptionsImpl((Pointer)this).offsetAddress(i);
    }


  /** Creates and returns a copy of this SessionOptions object. Wraps OrtApi::CloneSessionOptions */
  public native @ByVal SessionOptions Clone();

  /** Wraps OrtApi::GetSessionConfigEntry */
  public native @StdString BytePointer GetConfigEntry(@Cast("const char*") BytePointer config_key);
  public native @StdString String GetConfigEntry(String config_key);
  /** Wraps OrtApi::HasSessionConfigEntry */
  public native @Cast("bool") boolean HasConfigEntry(@Cast("const char*") BytePointer config_key);
  public native @Cast("bool") boolean HasConfigEntry(String config_key);
  public native @StdString BytePointer GetConfigEntryOrDefault(@Cast("const char*") BytePointer config_key, @StdString BytePointer def);
  public native @StdString String GetConfigEntryOrDefault(String config_key, @StdString String def);
}
