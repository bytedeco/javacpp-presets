// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tensorflowlite;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;

/** SignatureRunner class for running TFLite models using SignatureDef.
 * 
 *  Usage:
 * 
 *  <pre><code>
 *  // Create model from file. Note that the model instance must outlive the
 *  // interpreter instance.
 *  auto model = tflite::FlatBufferModel::BuildFromFile(...);
 *  if (model == nullptr) {
 *    // Return error.
 *  }
 * 
 *  // Create an Interpreter with an InterpreterBuilder.
 *  std::unique_ptr<tflite::Interpreter> interpreter;
 *  tflite::ops::builtin::BuiltinOpResolver resolver;
 *  if (InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
 *    // Return failure.
 *  }
 * 
 *  // Get the list of signatures and check it.
 *  auto signature_defs = interpreter->signature_keys();
 *  if (signature_defs.empty()) {
 *    // Return error.
 *  }
 * 
 *  // Get pointer to the SignatureRunner instance corresponding to a signature.
 *  // Note that the pointed SignatureRunner instance has lifetime same as the
 *  // Interpreter instance.
 *  tflite::SignatureRunner* runner =
 *                 interpreter->GetSignatureRunner(signature_defs[0]->c_str());
 *  if (runner == nullptr) {
 *    // Return error.
 *  }
 *  if (runner->AllocateTensors() != kTfLiteOk) {
 *    // Return failure.
 *  }
 * 
 *  // Set input data. In this example, the input tensor has float type.
 *  float* input = runner->input_tensor(0)->data.f;
 *  for (int i = 0; i < input_size; i++) {
 *    input[i] = ...; */
//  }
/** runner->Invoke();
/** </code></pre>
/**
/** WARNING: This class is *not* thread-safe. The client is responsible for
/** ensuring serialized interaction to avoid data races and undefined behavior.
/**
/** SignatureRunner and Interpreter share the same underlying data. Calling
/** methods on an Interpreter object will affect the state in corresponding
/** SignatureRunner objects. Therefore, it is recommended not to call other
/** Interpreter methods after calling GetSignatureRunner to create
/** SignatureRunner instances. */
@Namespace("tflite::impl") @NoOffset @Properties(inherit = org.bytedeco.tensorflowlite.presets.tensorflowlite.class)
public class SignatureRunner extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SignatureRunner(Pointer p) { super(p); }

  /** Returns the key for the corresponding signature. */
  public native @StdString String signature_key();

  /** Returns the number of inputs. */
  public native @Cast("size_t") long input_size();

  /** Returns the number of outputs. */
  public native @Cast("size_t") long output_size();

  /** Read-only access to list of signature input names. */
  public native @Cast("const char**") @StdVector PointerPointer input_names();

  /** Read-only access to list of signature output names. */
  public native @Cast("const char**") @StdVector PointerPointer output_names();

  /** Returns the input tensor identified by 'input_name' in the
   *  given signature. Returns nullptr if the given name is not valid. */
  public native TfLiteTensor input_tensor(@Cast("const char*") BytePointer input_name);
  public native TfLiteTensor input_tensor(String input_name);

  /** Returns the output tensor identified by 'output_name' in the
   *  given signature. Returns nullptr if the given name is not valid. */
  public native @Const TfLiteTensor output_tensor(@Cast("const char*") BytePointer output_name);
  public native @Const TfLiteTensor output_tensor(String output_name);

  /** Change a dimensionality of a given tensor. Note, this is only acceptable
   *  for tensors that are inputs.
   *  Returns status of failure or success. Note that this doesn't actually
   *  resize any existing buffers. A call to AllocateTensors() is required to
   *  change the tensor input buffer. */
  
  ///
  ///
  public native @Cast("TfLiteStatus") int ResizeInputTensor(@Cast("const char*") BytePointer input_name,
                                   @StdVector IntPointer new_size);
  public native @Cast("TfLiteStatus") int ResizeInputTensor(String input_name,
                                   @StdVector IntBuffer new_size);
  public native @Cast("TfLiteStatus") int ResizeInputTensor(@Cast("const char*") BytePointer input_name,
                                   @StdVector int[] new_size);
  public native @Cast("TfLiteStatus") int ResizeInputTensor(String input_name,
                                   @StdVector IntPointer new_size);
  public native @Cast("TfLiteStatus") int ResizeInputTensor(@Cast("const char*") BytePointer input_name,
                                   @StdVector IntBuffer new_size);
  public native @Cast("TfLiteStatus") int ResizeInputTensor(String input_name,
                                   @StdVector int[] new_size);

  /** Change the dimensionality of a given tensor. This is only acceptable for
   *  tensor indices that are inputs or variables.
   * 
   *  Difference from ResizeInputTensor: Only unknown dimensions can be resized
   *  with this function. Unknown dimensions are indicated as {@code -1} in the
   *  {@code dims_signature} attribute of a TfLiteTensor.
   * 
   *  Returns status of failure or success. Note that this doesn't actually
   *  resize any existing buffers. A call to AllocateTensors() is required to
   *  change the tensor input buffer. */
  public native @Cast("TfLiteStatus") int ResizeInputTensorStrict(@Cast("const char*") BytePointer input_name,
                                         @StdVector IntPointer new_size);
  public native @Cast("TfLiteStatus") int ResizeInputTensorStrict(String input_name,
                                         @StdVector IntBuffer new_size);
  public native @Cast("TfLiteStatus") int ResizeInputTensorStrict(@Cast("const char*") BytePointer input_name,
                                         @StdVector int[] new_size);
  public native @Cast("TfLiteStatus") int ResizeInputTensorStrict(String input_name,
                                         @StdVector IntPointer new_size);
  public native @Cast("TfLiteStatus") int ResizeInputTensorStrict(@Cast("const char*") BytePointer input_name,
                                         @StdVector IntBuffer new_size);
  public native @Cast("TfLiteStatus") int ResizeInputTensorStrict(String input_name,
                                         @StdVector int[] new_size);

  /** Updates allocations for all tensors, related to the given signature. */
  public native @Cast("TfLiteStatus") int AllocateTensors();

  /** Invokes the signature runner (run the graph identified by the given
   *  signature in dependency order). */
  public native @Cast("TfLiteStatus") int Invoke();

  /** Attempts to cancel in flight invocation if any.
   *  This will not affect calls to {@code Invoke} that happened after this.
   *  Non blocking and thread safe.
   *  Returns kTfLiteError if cancellation is not enabled, otherwise returns
   *  kTfLiteOk.
   *  WARNING: This is an experimental API and subject to change. */
  
  ///
  ///
  public native @Cast("TfLiteStatus") int Cancel();

  /** \brief Assigns (or reassigns) a custom memory allocation for the given
   *  tensor name. {@code flags} is a bitmask, see TfLiteCustomAllocationFlags.
   *  The runtime does NOT take ownership of the underlying memory.
   * 
   *  NOTE: User needs to call AllocateTensors() after this.
   *  Invalid/insufficient buffers will cause an error during AllocateTensors or
   *  Invoke (in case of dynamic shapes in the graph).
   * 
   *  Parameters should satisfy the following conditions:
   *  1. tensor->allocation_type == kTfLiteArenaRw or kTfLiteArenaRwPersistent
   *     In general, this is true for I/O tensors & variable tensors.
   *  2. allocation->data has the appropriate permissions for runtime access
   *     (Read-only for inputs, Read-Write for others), and outlives
   *     Interpreter.
   *  3. allocation->bytes >= tensor->bytes.
   *     This condition is checked again if any tensors are resized.
   *  4. allocation->data should be aligned to kDefaultTensorAlignment
   *     defined in lite/util.h. (Currently 64 bytes)
   *     This check is skipped if kTfLiteCustomAllocationFlagsSkipAlignCheck is
   *     set through {@code flags}.
   *  \warning This is an experimental API and subject to change. \n */
  
  ///
  ///
  public native @Cast("TfLiteStatus") int SetCustomAllocationForInputTensor(
        @Cast("const char*") BytePointer input_name, @Const @ByRef TfLiteCustomAllocation allocation,
        @Cast("int64_t") long flags/*=kTfLiteCustomAllocationFlagsNone*/);
  public native @Cast("TfLiteStatus") int SetCustomAllocationForInputTensor(
        @Cast("const char*") BytePointer input_name, @Const @ByRef TfLiteCustomAllocation allocation);
  public native @Cast("TfLiteStatus") int SetCustomAllocationForInputTensor(
        String input_name, @Const @ByRef TfLiteCustomAllocation allocation,
        @Cast("int64_t") long flags/*=kTfLiteCustomAllocationFlagsNone*/);
  public native @Cast("TfLiteStatus") int SetCustomAllocationForInputTensor(
        String input_name, @Const @ByRef TfLiteCustomAllocation allocation);

  /** \brief Assigns (or reassigns) a custom memory allocation for the given
   *  tensor name. {@code flags} is a bitmask, see TfLiteCustomAllocationFlags.
   *  The runtime does NOT take ownership of the underlying memory.
   * 
   *  NOTE: User needs to call AllocateTensors() after this.
   *  Invalid/insufficient buffers will cause an error during AllocateTensors or
   *  Invoke (in case of dynamic shapes in the graph).
   * 
   *  Parameters should satisfy the following conditions:
   *  1. tensor->allocation_type == kTfLiteArenaRw or kTfLiteArenaRwPersistent
   *     In general, this is true for I/O tensors & variable tensors.
   *  2. allocation->data has the appropriate permissions for runtime access
   *     (Read-only for inputs, Read-Write for others), and outlives
   *     Interpreter.
   *  3. allocation->bytes >= tensor->bytes.
   *     This condition is checked again if any tensors are resized.
   *  4. allocation->data should be aligned to kDefaultTensorAlignment
   *     defined in lite/util.h. (Currently 64 bytes)
   *     This check is skipped if kTfLiteCustomAllocationFlagsSkipAlignCheck is
   *     set through {@code flags}.
   *  \warning This is an experimental API and subject to change. \n */
  
  ///
  public native @Cast("TfLiteStatus") int SetCustomAllocationForOutputTensor(
        @Cast("const char*") BytePointer output_name, @Const @ByRef TfLiteCustomAllocation allocation,
        @Cast("int64_t") long flags/*=kTfLiteCustomAllocationFlagsNone*/);
  public native @Cast("TfLiteStatus") int SetCustomAllocationForOutputTensor(
        @Cast("const char*") BytePointer output_name, @Const @ByRef TfLiteCustomAllocation allocation);
  public native @Cast("TfLiteStatus") int SetCustomAllocationForOutputTensor(
        String output_name, @Const @ByRef TfLiteCustomAllocation allocation,
        @Cast("int64_t") long flags/*=kTfLiteCustomAllocationFlagsNone*/);
  public native @Cast("TfLiteStatus") int SetCustomAllocationForOutputTensor(
        String output_name, @Const @ByRef TfLiteCustomAllocation allocation);

  /** \brief Set if buffer handle output is allowed.
   * 
   *  When using hardware delegation, Interpreter will make the data of output
   *  tensors available in {@code tensor->data} by default. If the application can
   *  consume the buffer handle directly (e.g. reading output from OpenGL
   *  texture), it can set this flag to true, so Interpreter won't copy the
   *  data from buffer handle to CPU memory.
   *  \warning This is an experimental API and subject to change. \n */
  public native void SetAllowBufferHandleOutput(@Cast("bool") boolean allow_buffer_handle_output);

  /** \warning This is an experimental API and subject to change. \n
   *  \brief Set the delegate buffer handle to a input tensor.
   *  TfLiteDelegate should be aware of how to handle the buffer handle.
   *  {@code release_existing_buffer_handle}: If true, the existing buffer handle */
  // will be released by TfLiteDelegate::FreeBufferHandle.
  public native @Cast("TfLiteStatus") int SetInputBufferHandle(@Cast("const char*") BytePointer input_name,
                                      @Cast("TfLiteBufferHandle") int buffer_handle,
                                      TfLiteDelegate delegate,
                                      @Cast("bool") boolean release_existing_buffer_handle/*=true*/);
  public native @Cast("TfLiteStatus") int SetInputBufferHandle(@Cast("const char*") BytePointer input_name,
                                      @Cast("TfLiteBufferHandle") int buffer_handle,
                                      TfLiteDelegate delegate);
  public native @Cast("TfLiteStatus") int SetInputBufferHandle(String input_name,
                                      @Cast("TfLiteBufferHandle") int buffer_handle,
                                      TfLiteDelegate delegate,
                                      @Cast("bool") boolean release_existing_buffer_handle/*=true*/);
  public native @Cast("TfLiteStatus") int SetInputBufferHandle(String input_name,
                                      @Cast("TfLiteBufferHandle") int buffer_handle,
                                      TfLiteDelegate delegate);

  /** \warning This is an experimental API and subject to change. \n
   *  \brief Set the delegate buffer handle to a output tensor.
   *  TfLiteDelegate should be aware of how to handle the buffer handle.
   *  {@code release_existing_buffer_handle}: If true, the existing buffer handle
   *  will be released by TfLiteDelegate::FreeBufferHandle. */
  public native @Cast("TfLiteStatus") int SetOutputBufferHandle(
        @Cast("const char*") BytePointer output_name, @Cast("TfLiteBufferHandle") int buffer_handle,
        TfLiteDelegate delegate, @Cast("bool") boolean release_existing_buffer_handle/*=true*/);
  public native @Cast("TfLiteStatus") int SetOutputBufferHandle(
        @Cast("const char*") BytePointer output_name, @Cast("TfLiteBufferHandle") int buffer_handle,
        TfLiteDelegate delegate);
  public native @Cast("TfLiteStatus") int SetOutputBufferHandle(
        String output_name, @Cast("TfLiteBufferHandle") int buffer_handle,
        TfLiteDelegate delegate, @Cast("bool") boolean release_existing_buffer_handle/*=true*/);
  public native @Cast("TfLiteStatus") int SetOutputBufferHandle(
        String output_name, @Cast("TfLiteBufferHandle") int buffer_handle,
        TfLiteDelegate delegate);
}
