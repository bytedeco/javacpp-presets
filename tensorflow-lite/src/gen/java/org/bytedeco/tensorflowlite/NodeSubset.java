// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tensorflowlite;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;


// Represents a subset of nodes in a TensorFlow Lite graph.
@Namespace("tflite") @Properties(inherit = org.bytedeco.tensorflowlite.presets.tensorflowlite.class)
public class NodeSubset extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NodeSubset() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public NodeSubset(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NodeSubset(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public NodeSubset position(long position) {
        return (NodeSubset)super.position(position);
    }
    @Override public NodeSubset getPointer(long i) {
        return new NodeSubset((Pointer)this).offsetAddress(i);
    }

  /** enum tflite::NodeSubset::Type */
  public static final int
    kTfUnexplored = 0,  // temporarily used during creation
    kTfPartition = 1,
    kTfNonPartition = 2;
  public native @Cast("tflite::NodeSubset::Type") int type(); public native NodeSubset type(int setter);
  // Nodes within the node sub set
  public native @StdVector IntPointer nodes(); public native NodeSubset nodes(IntPointer setter);
  // Tensors that stride output from another node sub set that this depends on,
  // or global inputs to the TensorFlow Lite full graph.
  public native @StdVector IntPointer input_tensors(); public native NodeSubset input_tensors(IntPointer setter);
  // Outputs that are consumed by other node sub sets or are global output
  // tensors. All output tensors of the nodes in the node sub set that do not
  // appear in this list are intermediate results that can be potentially
  // elided.
  public native @StdVector IntPointer output_tensors(); public native NodeSubset output_tensors(IntPointer setter);
}
