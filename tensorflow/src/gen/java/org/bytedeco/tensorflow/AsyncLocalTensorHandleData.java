// Targeted by JavaCPP version 1.5.3: DO NOT EDIT THIS FILE

package org.bytedeco.tensorflow;

import org.bytedeco.tensorflow.Allocator;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.tensorflow.global.tensorflow.*;


// Async Local Tensor Handle: A non-ready local tensor handle used in async
// eager execution. Once the execution is complete this is replaced by a local
// tensor handle.
@Namespace("tensorflow") @Properties(inherit = org.bytedeco.tensorflow.presets.tensorflow.class)
public class AsyncLocalTensorHandleData extends TensorHandleData {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AsyncLocalTensorHandleData(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public AsyncLocalTensorHandleData(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public AsyncLocalTensorHandleData position(long position) {
        return (AsyncLocalTensorHandleData)super.position(position);
    }

  public AsyncLocalTensorHandleData() { super((Pointer)null); allocate(); }
  private native void allocate();

  // Async tensor handles are not ready and hence cannot satisfy any of these
  // requests.
  public native @ByVal Status Tensor(@Cast("const tensorflow::Tensor**") PointerPointer t);
  public native @ByVal Status Tensor(@Const @ByPtrPtr Tensor t);
  public native @ByVal Status TensorValue(TensorValue t);
  public native @ByVal Status Shape(TensorShape shape);
  public native @ByVal Status NumDims(IntPointer num_dims);
  public native @ByVal Status NumDims(IntBuffer num_dims);
  public native @ByVal Status NumDims(int... num_dims);
  public native @ByVal Status Dim(int dim_index, @Cast("tensorflow::int64*") LongPointer dim);
  public native @ByVal Status Dim(int dim_index, @Cast("tensorflow::int64*") LongBuffer dim);
  public native @ByVal Status Dim(int dim_index, @Cast("tensorflow::int64*") long... dim);
  public native @ByVal Status NumElements(@Cast("tensorflow::int64*") LongPointer num_elements);
  public native @ByVal Status NumElements(@Cast("tensorflow::int64*") LongBuffer num_elements);
  public native @ByVal Status NumElements(@Cast("tensorflow::int64*") long... num_elements);

  public native @StdString BytePointer DebugString();
}