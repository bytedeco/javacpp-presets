// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tensorflowlite;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;

// #endif  // TF_LITE_STATIC_MEMORY

/** Light-weight tensor struct for TF Micro runtime. Provides the minimal amount
 *  of information required for a kernel to run during TfLiteRegistration::Eval. */
// TODO(b/160955687): Move this field into TF_LITE_STATIC_MEMORY when TFLM
// builds with this flag by default internally.
@Properties(inherit = org.bytedeco.tensorflowlite.presets.tensorflowlite.class)
public class TfLiteEvalTensor extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public TfLiteEvalTensor() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public TfLiteEvalTensor(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TfLiteEvalTensor(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public TfLiteEvalTensor position(long position) {
        return (TfLiteEvalTensor)super.position(position);
    }
    @Override public TfLiteEvalTensor getPointer(long i) {
        return new TfLiteEvalTensor((Pointer)this).offsetAddress(i);
    }

  /** A union of data pointers. The appropriate type should be used for a typed
   *  tensor based on {@code type}. */
  public native @ByRef TfLitePtrUnion data(); public native TfLiteEvalTensor data(TfLitePtrUnion setter);

  /** A pointer to a structure representing the dimensionality interpretation
   *  that the buffer should have. */
  public native TfLiteIntArray dims(); public native TfLiteEvalTensor dims(TfLiteIntArray setter);

  /** The data type specification for data stored in {@code data}. This affects
   *  what member of {@code data} union should be used. */
  public native @Cast("TfLiteType") int type(); public native TfLiteEvalTensor type(int setter);
}
