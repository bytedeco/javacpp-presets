// Targeted by JavaCPP version 1.5.9-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.tritonserver.tritonserver;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tritonserver.global.tritonserver.*;


//==============================================================================
/** An interface for InferResult object to interpret the response to an
 *  inference request.
 *  */
@Namespace("triton::developer_tools::server") @Properties(inherit = org.bytedeco.tritonserver.presets.tritonserver.class)
public class GenericInferResult extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GenericInferResult(Pointer p) { super(p); }


  /** Get the name of the model which generated this response.
   *  @return Returns the name of the model. */
  public native @StdString @NoException(true) BytePointer ModelName();

  /** Get the version of the model which generated this response.
   *  @return Returns the version of the model. */
  public native @StdString @NoException(true) BytePointer ModelVersion();

  /** Get the id of the request which generated this response.
   *  @return Returns the id of the request. */
  public native @StdString @NoException(true) BytePointer Id();

  /** Get the output names from the infer result
   *  @return Vector of output names */
  public native @ByVal StringVector OutputNames();

  /** Get the result output as a shared pointer of 'Tensor' object. The 'buffer'
   *  field of the output is owned by the returned 'Tensor' object itself. Note
   *  that for string data, need to use 'StringData' function for string data
   *  result.
   *  @param name The name of the output tensor to be retrieved.
   *  @return Returns the output result as a shared pointer of 'Tensor' object. */
  public native @SharedPtr Tensor Output(@StdString BytePointer name);
  public native @SharedPtr Tensor Output(@StdString String name);

  /** Get the result data as a vector of strings. The vector will
   *  receive a copy of result data. An exception will be thrown if
   *  the data type of output is not 'BYTES'.
   *  @param output_name The name of the output to get result data.
   *  @return Returns the result data represented as a vector of strings. The
   *  strings are stored in the row-major order. */
  public native @ByVal StringVector StringData(
        @StdString BytePointer output_name);
  public native @ByVal StringVector StringData(
        @StdString String output_name);

  /** Return the complete response as a user friendly string.
   *  @return The string describing the complete response. */
  public native @StdString BytePointer DebugString();

  /** Return if there is an error within this result.
   *  @return True if this 'GenericInferResult' object has an error, false if no
   *  error. */
  public native @Cast("bool") boolean HasError();

  /** Return the error message of the error.
   *  @return The messsage for the error. Empty if no error. */
  public native @StdString BytePointer ErrorMsg();
}