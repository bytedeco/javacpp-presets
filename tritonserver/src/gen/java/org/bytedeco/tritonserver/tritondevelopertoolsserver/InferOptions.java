// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tritonserver.tritondevelopertoolsserver;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tritonserver.global.tritondevelopertoolsserver.*;


//==============================================================================
/** Structure to hold options for Inference Request.
 *  */
@Namespace("triton::developer_tools::server") @NoOffset @Properties(inherit = org.bytedeco.tritonserver.presets.tritondevelopertoolsserver.class)
public class InferOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public InferOptions(Pointer p) { super(p); }

  public InferOptions(@StdString BytePointer model_name) { super((Pointer)null); allocate(model_name); }
  private native void allocate(@StdString BytePointer model_name);
  public InferOptions(@StdString String model_name) { super((Pointer)null); allocate(model_name); }
  private native void allocate(@StdString String model_name);

  public InferOptions(
        @StdString BytePointer model_name, @Cast("const int64_t") long model_version,
        @StdString BytePointer request_id, @Cast("const uint64_t") long correlation_id,
        @StdString BytePointer correlation_id_str, @Cast("const bool") boolean sequence_start,
        @Cast("const bool") boolean sequence_end, @Cast("const uint64_t") long priority,
        @Cast("const uint64_t") long request_timeout,
        @SharedPtr Allocator custom_allocator,
        @SharedPtr Trace trace) { super((Pointer)null); allocate(model_name, model_version, request_id, correlation_id, correlation_id_str, sequence_start, sequence_end, priority, request_timeout, custom_allocator, trace); }
  private native void allocate(
        @StdString BytePointer model_name, @Cast("const int64_t") long model_version,
        @StdString BytePointer request_id, @Cast("const uint64_t") long correlation_id,
        @StdString BytePointer correlation_id_str, @Cast("const bool") boolean sequence_start,
        @Cast("const bool") boolean sequence_end, @Cast("const uint64_t") long priority,
        @Cast("const uint64_t") long request_timeout,
        @SharedPtr Allocator custom_allocator,
        @SharedPtr Trace trace);
  public InferOptions(
        @StdString String model_name, @Cast("const int64_t") long model_version,
        @StdString String request_id, @Cast("const uint64_t") long correlation_id,
        @StdString String correlation_id_str, @Cast("const bool") boolean sequence_start,
        @Cast("const bool") boolean sequence_end, @Cast("const uint64_t") long priority,
        @Cast("const uint64_t") long request_timeout,
        @SharedPtr Allocator custom_allocator,
        @SharedPtr Trace trace) { super((Pointer)null); allocate(model_name, model_version, request_id, correlation_id, correlation_id_str, sequence_start, sequence_end, priority, request_timeout, custom_allocator, trace); }
  private native void allocate(
        @StdString String model_name, @Cast("const int64_t") long model_version,
        @StdString String request_id, @Cast("const uint64_t") long correlation_id,
        @StdString String correlation_id_str, @Cast("const bool") boolean sequence_start,
        @Cast("const bool") boolean sequence_end, @Cast("const uint64_t") long priority,
        @Cast("const uint64_t") long request_timeout,
        @SharedPtr Allocator custom_allocator,
        @SharedPtr Trace trace);

  // The name of the model to run inference.
  public native @StdString BytePointer model_name_(); public native InferOptions model_name_(BytePointer setter);
  // The version of the model to use while running inference. The default
  // value is "-1" which means the server will select the
  // version of the model based on its internal policy.
  public native @Cast("int64_t") long model_version_(); public native InferOptions model_version_(long setter);
  // An identifier for the request. If specified will be returned
  // in the response. Default value is an empty string which means no
  // request_id will be used.
  public native @StdString BytePointer request_id_(); public native InferOptions request_id_(BytePointer setter);
  // The correlation ID of the inference request to be an unsigned integer.
  // Should be used exclusively with 'correlation_id_str_'.
  // Default is 0, which indicates that the request has no correlation ID.
  public native @Cast("uint64_t") long correlation_id_(); public native InferOptions correlation_id_(long setter);
  // The correlation ID of the inference request to be a string.
  // Should be used exclusively with 'correlation_id_'.
  // Default value is "".
  public native @StdString BytePointer correlation_id_str_(); public native InferOptions correlation_id_str_(BytePointer setter);
  // Indicates whether the request being added marks the start of the
  // sequence. Default value is False. This argument is ignored if
  // 'sequence_id' is 0.
  public native @Cast("bool") boolean sequence_start_(); public native InferOptions sequence_start_(boolean setter);
  // Indicates whether the request being added marks the end of the
  // sequence. Default value is False. This argument is ignored if
  // 'sequence_id' is 0.
  public native @Cast("bool") boolean sequence_end_(); public native InferOptions sequence_end_(boolean setter);
  // Indicates the priority of the request. Priority value zero
  // indicates that the default priority level should be used
  // (i.e. same behavior as not specifying the priority parameter).
  // Lower value priorities indicate higher priority levels. Thus
  // the highest priority level is indicated by setting the parameter
  // to 1, the next highest is 2, etc. If not provided, the server
  // will handle the request using default setting for the model.
  public native @Cast("uint64_t") long priority_(); public native InferOptions priority_(long setter);
  // The timeout value for the request, in microseconds. If the request
  // cannot be completed within the time by the server can take a
  // model-specific action such as terminating the request. If not
  // provided, the server will handle the request using default setting
  // for the model.
  public native @Cast("uint64_t") long request_timeout_(); public native InferOptions request_timeout_(long setter);
  // User-provided custom reponse allocator object. Default is nullptr.
  // If using custom allocator, the lifetime of this 'Allocator' object should
  // be long enough until `InferResult` object goes out of scope as we need
  // this `Allocator` object to call 'ResponseAllocatorReleaseFn_t' for
  // releasing the response.
  public native @SharedPtr Allocator custom_allocator_(); public native InferOptions custom_allocator_(Allocator setter);
  // Update trace setting for the specified model. If not set, will use global
  // trace setting in 'ServerOptions' for tracing if tracing is enabled in
  // 'ServerOptions'. Default is nullptr.
  public native @SharedPtr Trace trace_(); public native InferOptions trace_(Trace setter);
}
